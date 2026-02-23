#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import tqdm

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from collections import deque

import utils
import hydra

import importlib
from omegaconf import DictConfig, ListConfig, open_dict  # add this near your other imports


def build_agent_from_cfg(agent_cfg: DictConfig):
    """
    agent_cfg is expected to look like:
      agent:
        name: sac
        class: agent.sac.SACAgent
        params: { ... }

    Here we:
      - import the class from `agent_cfg.class`
      - pass `agent_cfg.params` as **kwargs

    Nested fields like critic_cfg / actor_cfg stay as DictConfig,
    so SACAgent can still call hydra.utils.instantiate() on them.
    """
    cls_path = agent_cfg["class"]
    params_cfg = agent_cfg["params"]   # DictConfig

    module_name, class_name = cls_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    AgentCls = getattr(mod, class_name)

    # pass DictConfig directly; Hydra instantiate is not used here,
    # but SACAgent will use hydra.utils.instantiate on critic_cfg/actor_cfg.
    return AgentCls(**params_cfg)


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        # Robustly get agent name so logger works for all envs
        try:
            agent_name = cfg.agent.name
        except Exception:
            try:
                agent_name = cfg.agent.get("name", "sac")
            except Exception:
                agent_name = "sac"

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=agent_name,
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False

        # -------------------
        # 1) Make environment
        # -------------------
        if "metaworld" in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        # -------------------
        # 1b) Figure out max episode length in a robust way
        # -------------------
        if hasattr(self.env, "_max_episode_steps"):
            self.max_episode_steps = int(self.env._max_episode_steps)
        elif hasattr(self.env, "spec") and getattr(self.env.spec, "max_episode_steps", None) is not None:
            self.max_episode_steps = int(self.env.spec.max_episode_steps)
        else:
            self.max_episode_steps = int(getattr(cfg, "max_episode_steps", 365))

        # -------------------
        # 2) Infer obs/action dims & range
        # -------------------
        obs_dim = self.env.observation_space.shape[0]

        action_space = self.env.action_space
        if (
            hasattr(action_space, "shape")
            and action_space.shape is not None
            and len(action_space.shape) > 0
        ):
            # Continuous action space (e.g., HalfCheetah, DMC)
            action_dim = action_space.shape[0]
            action_range = [
                float(action_space.low.min()),
                float(action_space.high.max()),
            ]
            action_shape = action_space.shape
        else:
            # Discrete action space (e.g., WOFOST Discrete(17))
            action_dim = 1
            action_range = [0.0, float(action_space.n - 1)]
            action_shape = (action_dim,)

        # Write back into config for SACAgent, critic, actor, etc.
        cfg.agent.params.obs_dim = int(obs_dim)
        cfg.agent.params.action_dim = int(action_dim)
        cfg.agent.params.action_range = action_range

        # with open_dict(cfg):
        #     if "agent" not in cfg or cfg["agent"] is None:
        #         cfg["agent"] = {}
        #     if "params" not in cfg["agent"] or cfg["agent"]["params"] is None:
        #         cfg["agent"]["params"] = {}

        #     cfg["agent"]["params"]["obs_dim"] = int(obs_dim)
        #     cfg["agent"]["params"]["action_dim"] = int(action_dim)
        #     cfg["agent"]["params"]["action_range"] = action_range

        # -------------------
        # 3) Build agent
        # -------------------
        self.agent = build_agent_from_cfg(cfg.agent)

        # -------------------
        # 4) Replay buffer
        # -------------------
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            action_shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        # For logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # -------------------
        # 5) Reward model (Tier-2: WOFOST-only segment override)
        # -------------------
        size_segment = int(cfg.segment)

        if utils.is_wofost(cfg) and getattr(cfg, "wofost", None) is not None and cfg.wofost.enable_tier2:
            if cfg.wofost.segment_len_override is not None:
                size_segment = int(cfg.wofost.segment_len_override)
                print(f"[WOFOST Tier2] Overriding segment length: {cfg.segment} -> {size_segment}")

        self.reward_model = RewardModel(
            obs_dim,
            action_dim,
            ensemble_size=cfg.ensemble_size,
            size_segment=size_segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
        )

    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
                
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        # ----------------------------------------
        # Pref label distribution logging (per round)
        # ----------------------------------------
        hist = getattr(self.reward_model, "last_label_hist", None)
        if hist is not None:
            # Console print (easy sanity check)
            print(
                f"[pref] step={self.step} mb={self.reward_model.mb_size} "
                f"n={hist['n_total']} seg1={hist['prefer_seg1']} "
                f"seg2={hist['prefer_seg2']} tie={hist['tie']} "
                f"p1={hist['p_prefer_seg1']:.3f} p2={hist['p_prefer_seg2']:.3f} ptie={hist['p_tie']:.3f}"
            )

            # TensorBoard scalars
            self.logger.log("pref/n_total", hist["n_total"], self.step)
            self.logger.log("pref/n_prefer_seg1", hist["prefer_seg1"], self.step)
            self.logger.log("pref/n_prefer_seg2", hist["prefer_seg2"], self.step)
            self.logger.log("pref/n_tie", hist["tie"], self.step)

            self.logger.log("pref/p_prefer_seg1", hist["p_prefer_seg1"], self.step)
            self.logger.log("pref/p_prefer_seg2", hist["p_prefer_seg2"], self.step)
            self.logger.log("pref/p_tie", hist["p_tie"], self.step)
        
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;
                    
        # print("Reward function is updated!! ACC: " + str(total_acc))
        if self.labeled_feedback > 0:
            print("Reward function is updated!! ACC: " + str(total_acc))
        else:
            print("Reward function update skipped (no labeled feedback yet).")

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10) 
        start_time = time.time()

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)
                
                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.max_episode_steps)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.max_episode_steps)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, done, extra = self.env.step(action)
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        
@hydra.main(config_path='config', config_name='train_PEBBLE')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
