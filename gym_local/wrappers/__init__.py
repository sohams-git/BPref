"""
Shim package so that `from gym.wrappers...` works by forwarding to gymnasium.wrappers.
"""

from gymnasium.wrappers import *  # re-export all wrappers
