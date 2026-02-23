"""
Shim module so that `import gym` in BPref works by forwarding to gymnasium.

We only need minimal functionality (spaces, basic wrappers) because WOFOSTGym
uses gymnasium directly and we won't be using BPref's MuJoCo envs here.
"""

from gymnasium import *
from gymnasium import spaces
