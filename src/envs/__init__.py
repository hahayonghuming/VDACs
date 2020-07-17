from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from envs.smart_man_sim.smart_man import SmartEnv
import sys
import os

"""
This script registers the multi-agents environment that we need to to test on
"""
def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["smart_man"] = partial(env_fn, env=SmartEnv)
REGISTRY["smart_man_flat"] = partial(env_fn, env=SmartEnv)
#TODO I need to register my environment here
# REGISTRY["sman"] = partial(env_fn, env=StarCraft2Env)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
