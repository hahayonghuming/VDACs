from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner

from .policy_gradient_v1 import PGLearner_v1
from .policy_gradient_v2 import PGLearner_v2
from .policy_gradient_v3 import PGLearner_v3


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner

REGISTRY["policy_gradient_v1"] = PGLearner_v1
REGISTRY["policy_gradient_v2"] = PGLearner_v2
REGISTRY["policy_gradient_v3"] = PGLearner_v3

