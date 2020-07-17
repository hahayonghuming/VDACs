REGISTRY = {}

from .rnn_agent import RNNAgent, RNNPPOAgent
from .gcn_agent import TRANSAgent, GATAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["transformer"] = TRANSAgent
REGISTRY["gat"] = GATAgent
