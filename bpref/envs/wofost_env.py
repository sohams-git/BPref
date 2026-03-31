from __future__ import annotations

import os
from typing import Optional
import pcse_gym
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig

from pcse_gym.args import NPK_Args, WOFOST_Args, Agro_Args

from dataclasses import fields
import inspect
# Root of your WOFOSTGym repo
WOFOST_ROOT = "/home/sohams/wofost/wofostgymhub/WOFOSTGym-master"

# Path to the PPO config.yaml you used for lnpkw-v0
# WOFOST_CONFIG_PATH = os.path.join(
#     WOFOST_ROOT,
#     "logs/ppo/PPO/lnpkw-v0__rl_utils__1__1763133702/config.yaml",
# )

def get_config_path(env_id: str) -> str:
    ppo_root = os.path.join(WOFOST_ROOT, "logs/ppo/PPO")
    if not os.path.isdir(ppo_root):
        raise FileNotFoundError(f"PPO log dir not found: {ppo_root}")

    matches = []
    for d in os.listdir(ppo_root):
        full = os.path.join(ppo_root, d)
        cfg = os.path.join(full, "config.yaml")
        if d.startswith(env_id) and os.path.isfile(cfg):
            matches.append((os.path.getmtime(cfg), cfg))

    if not matches:
        raise FileNotFoundError(f"No PPO config.yaml found for env_id={env_id} under {ppo_root}")

    matches.sort(reverse=True)
    return matches[0][1]


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
        # if not os.path.isfile(WOFOST_CONFIG_PATH):
        #     raise FileNotFoundError(
        #         f"WOFOST config.yaml not found at {WOFOST_CONFIG_PATH}. "
        #         f"Update WOFOST_CONFIG_PATH in bpref/envs/wofost_env.py if the run folder changed."
        #     )

        # conf = OmegaConf.load(WOFOST_CONFIG_PATH)
        config_path = get_config_path(env_id)
        print(f"[WOFOSTEnv] Using config: {config_path}")
        conf = OmegaConf.load(config_path)

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
        def _filter_to_dataclass(dc_cls, cfg_dict: dict):
            allowed = {f.name for f in fields(dc_cls)}
            return {k: v for k, v in cfg_dict.items() if k in allowed}
        # Convert wf/ag configs so things like CVL: [0.685] become CVL: 0.685
        # Convert wf/ag configs so things like CVL: [0.685] become CVL: 0.685
        wf_cfg = _unwrap_singleton_lists(dict(npk_cfg["wf"]))
        ag_cfg = _unwrap_singleton_lists(dict(npk_cfg["ag"]))

        # Filter out keys that are not fields of the dataclasses (e.g., site_name)
        wf_cfg_f = _filter_to_dataclass(WOFOST_Args, wf_cfg)
        ag_cfg_f = _filter_to_dataclass(Agro_Args, ag_cfg)

        dropped_ag = sorted(list(set(ag_cfg.keys()) - set(ag_cfg_f.keys())))
        if len(dropped_ag) > 0:
            print(f"[WOFOSTEnv] Dropping unknown Agro_Args keys: {dropped_ag}")

        wf = WOFOST_Args(**wf_cfg_f)
        ag = Agro_Args(**ag_cfg_f)
        npk_args = NPK_Args(wf=wf, ag=ag)

        # ---- 2. Base path and config paths ----
        base_fpath = WOFOST_ROOT + "/"

        def get_conf(key: str, default: str) -> str:
            return str(conf.get(key, default))

        agro_fpath = get_conf("agro_fpath", "env_config/agro/")
        agro_file = str(conf.get("agro_file", "wheat_agro.yaml"))

        site_fpath = get_conf("site_fpath", "env_config/site/")
        crop_fpath = get_conf("crop_fpath", "env_config/crop/")
        soil_fpath = get_conf("soil_fpath", "env_config/soil/")
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
        soil_fpath = to_relative(soil_fpath)
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
        # ---- 3. Create the underlying WOFOST Gymnasium env ----
        # Build full kwargs, then filter by what the env actually accepts.
        make_kwargs = dict(
            args=npk_args,
            base_fpath=base_fpath,
            agro_fpath=agro_fpath,
            soil_fpath=soil_fpath,
            site_fpath=site_fpath,
            crop_fpath=crop_fpath,
            name_fpath=name_fpath,
            unit_fpath=unit_fpath,
            range_fpath=range_fpath,
            render_mode=render_mode,
        )

        # Introspect env constructor signature and drop unsupported kwargs
        env_cls = gym.spec(env_id_effective).entry_point
        # entry_point can be "module:Class" or a callable; gymnasium resolves it internally,
        # but we can still filter using a trial signature approach:
        try:
            # Try to resolve class from entry_point string if needed
            if isinstance(env_cls, str) and ":" in env_cls:
                mod_name, cls_name = env_cls.split(":")
                mod = __import__(mod_name, fromlist=[cls_name])
                env_ctor = getattr(mod, cls_name)
            else:
                env_ctor = env_cls

            sig = inspect.signature(env_ctor.__init__)
            allowed = set(sig.parameters.keys())
            # remove "self"
            allowed.discard("self")

            filtered_kwargs = {k: v for k, v in make_kwargs.items() if k in allowed}

            dropped = sorted([k for k in make_kwargs.keys() if k not in allowed])
            if dropped:
                print(f"[WOFOSTEnv] Dropping unsupported env kwargs: {dropped}")

        except Exception as e:
            # Fallback: if introspection fails, at least drop the known troublemakers
            print(f"[WOFOSTEnv] Warning: could not introspect env signature ({e}). Using minimal kwargs.")
            filtered_kwargs = dict(args=npk_args, base_fpath=base_fpath, agro_fpath=agro_fpath, render_mode=render_mode)

        self._env = gym.make(env_id_effective, **filtered_kwargs)

        # ---- 4a. Cache underlying obs bounds (if finite) ----
        base_obs_space = self._env.observation_space
        self._obs_low = None
        self._obs_high = None

        if isinstance(base_obs_space, spaces.Box):
            low = np.asarray(base_obs_space.low)
            high = np.asarray(base_obs_space.high)

            # cast to float32
            low32 = low.astype(np.float32, copy=False)
            high32 = high.astype(np.float32, copy=False)

            # only use if bounds are finite
            if np.all(np.isfinite(low32)) and np.all(np.isfinite(high32)):
                self._obs_low = low32
                self._obs_high = high32

        # ---- 4b. Precompute denom for [-1,1] normalization ----
        self._obs_denom = None
        if self._obs_low is not None and self._obs_high is not None:
            denom = (self._obs_high - self._obs_low).astype(np.float32, copy=False)
            denom = np.where(denom == 0.0, 1.0, denom)  # avoid div-by-zero
            self._obs_denom = denom

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
        # obs_shape = self._env.observation_space.shape
        # self.observation_space = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=obs_shape,
        #     dtype=np.float32,
        # )

        obs_shape = self._env.observation_space.shape

        if self._obs_low is not None and self._obs_high is not None:
            self.observation_space = spaces.Box(
                low=self._obs_low,
                high=self._obs_high,
                shape=obs_shape,
                dtype=np.float32,
            )
        else:
            BIG = 1e6
            self.observation_space = spaces.Box(
                low=-BIG,
                high=BIG,
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
    def _cast_and_clip_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)

        if self._obs_low is not None and self._obs_high is not None and self._obs_denom is not None:
            # 1) clip to env bounds
            obs = np.clip(obs, self._obs_low, self._obs_high)

            # 2) normalize to [-1, 1]
            obs = 2.0 * (obs - self._obs_low) / self._obs_denom - 1.0

            # 3) guard for numeric noise
            obs = np.clip(obs, -1.0, 1.0)
        else:
            # fallback: prevent crazy magnitudes
            obs = np.clip(obs, -1e6, 1e6)

        return obs
    # ---- Old-Gym style API expected by PEBBLE ----
    def reset(self):
        out = self._env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return self._cast_and_clip_obs(obs)

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

        obs = self._cast_and_clip_obs(obs)
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