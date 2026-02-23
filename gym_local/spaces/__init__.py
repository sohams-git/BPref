"""
Shim so that `from gym.spaces import ...` works by forwarding to gymnasium.spaces.
"""

from gymnasium.spaces import *  # re-export all spaces (Box, Discrete, etc.)
