from .gym_halfcheetah import GymHalfCheetah

def make_env(env_name, **kwargs):
    if env_name == "gym-halfcheetah-v2":
        return GymHalfCheetah(**kwargs)
    raise ValueError(f"Unknown env_name: {env_name}")
