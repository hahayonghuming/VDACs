from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os

"""
This script registers the multi-agents environment that we need to to test on
"""
def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
