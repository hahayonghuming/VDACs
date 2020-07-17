import torch as th
import torch.nn as nn
import torch.nn.functional as F
import copy


def onehot_encode(tensor, target_dim):
    y_onehot = tensor.new(*tensor.shape[:-1], target_dim).zero_()
    y_onehot.scatter_(-1, tensor.long(), 1)
    return y_onehot.float()

class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, batch, action=None):
        inputs = self._build_inputs(batch, actions=action)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, actions=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        inputs = []

        # states repeated for all agents
        inputs.append(batch["state"][:, :].unsqueeze(2).repeat(1, 1, self.n_agents, 1).view(bs, max_t, self.n_agents, -1))

        # last actions
        if actions == None:
            actions = batch["actions_onehot"][:, :]  #[batch, time, agent, actions]
            overall_actions = actions.clone().detach()
        else:
            # actions are of shape [batch, time, agent, 1], first we need to onehot code it
            if actions.size()[-1] == 1:
                actions = onehot_encode(actions, self.n_actions)
                overall_actions = actions.clone().detach()

        overall_actions = overall_actions[:, :].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        overall_mask = (1 - th.eye(self.n_agents, device=batch.device))
        overall_mask = overall_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        overall_actions = overall_actions * overall_mask.unsqueeze(0).unsqueeze(0)
        # inputs.append(overall_actions * agent_mask.unsqueeze(0).unsqueeze(0))

        agent_actions = actions[:, :].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        agent_actions = agent_actions * agent_mask.unsqueeze(0).unsqueeze(0)

        tot_actions = overall_actions + agent_actions

        inputs.append(tot_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # actions masked out one agent
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # actions of the agent
        # input_shape += scheme["actions_onehot"]["vshape"][0]
        # agent id
        input_shape += self.n_agents
        return input_shape
