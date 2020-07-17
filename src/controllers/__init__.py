REGISTRY = {}

from .basic_controller import BasicMAC
from .ppo_controller import PPOMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["ppo_mac"] = PPOMAC
