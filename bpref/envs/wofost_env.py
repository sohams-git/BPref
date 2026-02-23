from __future__ import annotations

import os
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig

from pcse_gym.args import NPK_Args, WOFOST_Args, Agro_Args

# Root of your WOFOSTGym repo
WOFOST_ROOT = "/home/sohams/wofost/wofostgymhub/WOFOSTGym-master"

# Path to the PPO config.yaml you used for lnpkw-v0
WOFOST_CONFIG_PATH = os.path.join(
    WOFOST_ROOT,
    "logs/ppo/PPO/lnpkw-v0__rl_utils__1__1763133702/config.yaml",
)


class WofostEnv:
    """
    Thin wrapper that:
      - builds a WOFOST Gym env using the PPO config.yaml (wf/ag/path setup),
      - exposes old-Gym style API: reset() -> obs, step(a) -> (obs, r, done, info),
      - exposes continuous Box action space so SAC/PEBBLE see a standard env.
    """

    def __init__(
        self,
        env_id: str = "lnpkw-v0",
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        max_ep_len: Optional[int] = None,
        **_ignored,
    ):
        # ---- 1. Load WOFOST PPO config ----
        if not os.path.isfile(WOFOST_CONFIG_PATH):
            raise FileNotFoundError(
                f"WOFOST config.yaml not found at {WOFOST_CONFIG_PATH}. "
                f"Update WOFOST_CONFIG_PATH in bpref/envs/wofost_env.py if the run folder changed."
            )

        conf = OmegaConf.load(WOFOST_CONFIG_PATH)

        # Extract nested npk.wf and npk.ag and build proper dataclasses
        npk_cfg = conf["npk"]

        # --- helper: convert ListConfig([x]) or [x] -> float(x) ---
        def _unwrap_singleton_lists(d: dict):
            out = {}
            for k, v in d.items():
                # If value is OmegaConf ListConfig or plain list with
                # exactly one numeric element, turn it into a float.
                if isinstance(v, (ListConfig, list)) and len(v) == 1 and isinstance(v[0], (int, float)):
                    out[k] = float(v[0])
                else:
                    out[k] = v
            return out

        # Convert wf/ag configs so things like CVL: [0.685] become CVL: 0.685
        wf_cfg = _unwrap_singleton_lists(dict(npk_cfg["wf"]))
        ag_cfg = _unwrap_singleton_lists(dict(npk_cfg["ag"]))

        wf = WOFOST_Args(**wf_cfg)
        ag = Agro_Args(**ag_cfg)
        npk_args = NPK_Args(wf=wf, ag=ag)

        # ---- 2. Base path and config paths ----
        base_fpath = WOFOST_ROOT + "/"

        def get_conf(key: str, default: str) -> str:
            return str(conf.get(key, default))

        agro_fpath = get_conf("agro_fpath", "env_config/agro/")
        agro_file = str(conf.get("agro_file", "wheat_agro.yaml"))

        site_fpath = get_conf("site_fpath", "env_config/site/")
        crop_fpath = get_conf("crop_fpath", "env_config/crop/")
        unit_fpath = get_conf("unit_fpath", "env_config/state_units.yaml")
        name_fpath = get_conf("name_fpath", "env_config/state_names.yaml")
        range_fpath = get_conf("range_fpath", "env_config/state_ranges.yaml")
        render_mode_conf = conf.get("render_mode", None)

        def to_relative(p: str) -> str:
            if not os.path.isabs(p):
                return p

            root = WOFOST_ROOT.rstrip("/") + "/"
            if p.startswith(root):
                return p[len(root):]

            cfg_base = str(conf.get("base_fpath", "")).rstrip("/") + "/"
            if cfg_base != "/" and p.startswith(cfg_base):
                return p[len(cfg_base):]

            return p

        agro_fpath = to_relative(agro_fpath)
        site_fpath = to_relative(site_fpath)
        crop_fpath = to_relative(crop_fpath)
        unit_fpath = to_relative(unit_fpath)
        name_fpath = to_relative(name_fpath)
        range_fpath = to_relative(range_fpath)

        # If agro_fpath still looks like a directory, append agro_file
        if agro_fpath.endswith("/") or agro_fpath == "env_config/agro":
            agro_fpath = os.path.join(agro_fpath, agro_file)

        if render_mode is None:
            render_mode = render_mode_conf

        env_id_effective = env_id or conf.get("env_id", "lnpkw-v0")

        # ---- 3. Create the underlying WOFOST Gymnasium env ----
        self._env = gym.make(
            env_id_effective,
            args=npk_args,
            base_fpath=base_fpath,
            agro_fpath=agro_fpath,
            site_fpath=site_fpath,
            crop_fpath=crop_fpath,
            name_fpath=name_fpath,
            unit_fpath=unit_fpath,
            range_fpath=range_fpath,
            render_mode=render_mode,
        )

        # Seed if requested
        if seed is not None:
            try:
                self._env.reset(seed=seed)
            except TypeError:
                try:
                    self._env.seed(seed)
                except Exception:
                    pass

        # ---- 4. Expose "nice" spaces to PEBBLE/SAC ----
        obs_shape = self._env.observation_space.shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )

        # Underlying action space is Discrete(17). Expose Box(-1,1) for SAC.
        assert hasattr(self._env.action_space, "n"), "Underlying WOFOST env must be Discrete."
        self._num_discrete = self._env.action_space.n
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    # ---- Old-Gym style API expected by PEBBLE ----
    def reset(self):
        out = self._env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        obs = np.asarray(obs, dtype=np.float32)
        return obs

    def step(self, action):
        # action is continuous in [-1, 1], shape (1,)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a_scalar = float(np.clip(a[0], -1.0, 1.0))

        # Map [-1, 1] -> discrete {0, ..., n-1}
        idx = int((a_scalar + 1.0) / 2.0 * (self._num_discrete - 1) + 0.5)
        idx = int(np.clip(idx, 0, self._num_discrete - 1))

        out = self._env.step(idx)

        # Gymnasium style: (obs, reward, terminated, truncated, info)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = out

        obs = np.asarray(obs, dtype=np.float32)
        return obs, float(reward), bool(done), info

    def seed(self, s: int):
        try:
            self._env.reset(seed=s)
        except TypeError:
            try:
                self._env.seed(s)
            except Exception:
                pass

    def close(self):
        self._env.close()