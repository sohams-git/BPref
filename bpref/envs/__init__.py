# from .gym_halfcheetah import GymHalfCheetah
#  from .wofost_env import WofostEnv


# def make_env(env_name, **kwargs):
#     """
#     Low-level env factory used by utils.make_env for BPref-specific envs.
#     """
#     # HalfCheetah via Gym
#     if env_name.startswith("gym-halfcheetah"):
#         return GymHalfCheetah(env_id="HalfCheetah-v2", **kwargs)

#     # WOFOST branch
#     if env_name == "wofost-lnpkw-v0":
#         # internally uses env_id "lnpkw-v0" from pcse_gym
#         return WofostEnv(env_id="lnpkw-v0", **kwargs)

#     raise ValueError(f"Unknown env_name: {env_name}")

from .gym_mujoco import GymMujocoEnv
# from .wofost_env import WofostEnv

def make_env(env_name, **kwargs):
    """
    Low-level env factory used by utils.make_env for BPref-specific envs.
    """

    # Map BPref env_name -> Gym env_id
    GYM_MAP = {
        "gym-halfcheetah": "HalfCheetah-v2",
        "gym-hopper":      "Hopper-v2",
        "gym-walker2d":    "Walker2d-v2",
        "gym-swimmer":     "Swimmer-v2",
        "gym-ant":         "Ant-v2",
        "gym-humanoid":    "Humanoid-v2",
        "gym-reacher":     "Reacher-v2",
    }

    # allow prefixes like "gym-hopper" or "gym-hopper-v0" etc.
    for key, gym_id in GYM_MAP.items():
        if env_name.startswith(key):
            return GymMujocoEnv(env_id=gym_id, **kwargs)

    # WOFOST branch (optional)
    if env_name == "wofost-lnpkw-v0":
        from .wofost_env import WofostEnv
        return WofostEnv(env_id="lnpkw-v0", **kwargs)

    raise ValueError(f"Unknown env_name: {env_name}")