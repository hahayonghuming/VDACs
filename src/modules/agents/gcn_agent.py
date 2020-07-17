import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

"""
GAT args need to has the following properties:
- adj
- batch_size
- n_agents
- num_blocks
- attn_pdrop
- resid_pdrop
"""

class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = nn.Parameter(torch.tensor(e), requires_grad=False)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x

class GCN(nn.Module):
    def __init__(self, nf, nx, adj):
        super(GCN, self).__init__()
        self.nf = nf
        self.nx = nx
        self.adj = adj
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.w = Parameter(w)
        self.b = Parameter(torch.zeros(nf))

    def forward(self, x):
        pre = torch.matmul(x, self.w)
        x = torch.matmul(adj, pre)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_head, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % n_head == 0
        self.b = torch.tensor(cfg.adj, dtype=torch.float32, device=cfg.device)
        self.n_head = n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.residue = cfg.residue
        self.c_proj_1 = Conv1D(n_state, 1, nx)
        # self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        # self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        b = self.b
        if torch.sum(b) != np.prod(b.size()):
            w = w * b + -1e9 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        # w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        if self.residue:
            residue = query.clone()
            residue = self.merge_heads(residue)
            re = self.c_proj_1(residue)
            a += re
        # a = self.resid_dropout(a)
        return a


class GATAgent(nn.Module):
    """
    Graph attention network
    """
    def __init__(self, input_shape, args):
        super(GATAgent, self).__init__()
        self.args = args
        self.adj = torch.tensor(args.adj, dtype=torch.float32, device=args.device)
        self.bias = -1e9 * (1.0 - self.adj)
        self.residue = args.residue

        self.conv1 = Conv1D(args.rnn_hidden_dim, 1, input_shape)
        self.conv2 = Conv1D(1, 1, args.rnn_hidden_dim)
        self.conv3 = Conv1D(1, 1, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return Parameter(torch.zeros(1, self.args.rnn_hidden_dim, device=self.args.device))

    def forward(self, inputs, hidden_state):
        # x shape is [batch_size * n_agents, input_shape]
        # encode raw data into feature tensors
        inputs = inputs.view(self.args.batch_size, self.args.n_agents, -1)
        x = F.relu(self.conv1(inputs))
        # calculate attention weights
        f2 = self.conv2(x)
        f3 = self.conv3(x)
        logits = f2 + f3.transpose(-1, -2)
        logits *= self.adj
        # add bias to make the weights for non-connected nodes very small
        att_weights = nn.Softmax(dim=-1)(nn.LeakyReLU()(logits) + self.bias)
        att = torch.matmul(att_weights, x)
        if self.residue:
            att += x
        # reshape att from [batch_size, n_agents, rnn_hidden_dim] to [batch_size * n_agents, rnn_hidden_dim]
        att = att.view(-1, self.args.rnn_hidden_dim)
        # rnn learns the changes of node embeddings over time
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(att, h_in)
        q = self.fc1(h)
        return q, h


class GCNAgent(nn.Module):
    """
    Graph convolution network
    """
    def __init__(self, input_shape, args):
        super(GCNAgent, self).__init__()
        self.args = args
        self.adj = torch.tensor(args.adj, dtype=torch.float32, device=args.device)
        # self.bias = -1e9 * (1.0 - self.adj)

        self.gcn1 = GCN(args.rnn_hidden_dim, input_shape, self.adj)
        self.gcn2 = GCN(args.rnn_hidden_dim, args.rnn_hidden_dim, self.adj)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc1 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return Parameter(torch.zeros(1, self.args.rnn_hidden_dim, device=self.args.device))

    def forward(self, inputs, hidden_state):
        # x shape is [batch_size * n_agents, input_shape]
        # encode raw data into feature tensors
        inputs = inputs.view(self.args.batch_size, self.args.n_agents, -1)
        x = self.conv1(inputs)
        # calculate attention weights
        x = F.relu(self.gcn1(x))
        x = self.gcn2(x)
        # reshape att from [batch_size, n_agents, rnn_hidden_dim] to [batch_size * n_agents, rnn_hidden_dim]
        x = x.view(-1, self.args.rnn_hidden_dim)
        # rnn learns the changes of node embeddings over time
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc1(h)
        return q, h


class TRANSAgent(nn.Module):
    """
    Transformer network
    """
    def __init__(self, input_shape, args):
        super(TRANSAgent, self).__init__()
        self.args = args
        self.adj = torch.tensor(args.adj, dtype=torch.float32, device=args.device)
        num_blocks = args.num_blocks
        self.bias = -1e9 * (1.0 - self.adj)
        self.residue = args.residue

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.blocks = []
        for i in range(num_blocks):
            if args.device == 'cuda':
                self.blocks.append(Attention(args.rnn_hidden_dim, args.n_head, args).cuda())
            else:
                self.blocks.append(Attention(args.rnn_hidden_dim, args.n_head, args))
        self.ln = LayerNorm(args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(2 * args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return Parameter(torch.zeros(1, self.args.rnn_hidden_dim, device=self.args.device))

    def forward(self, inputs, hidden_state):
        # x shape is [batch_size * n_agents, input_shape]
        # encode raw data into feature tensors
        inputs = inputs.view(self.args.batch_size, self.args.n_agents, -1)
        x = F.relu(self.fc1(inputs))
        # x = self.fc1(inputs)
        embed = x
        # x size is [batch_size, n_agents, rnn_hidden_dim]
        for att in self.blocks:
            embed = att(embed)
        # embed = self.ln(embed)
        # message_x = embed
        message_x = self.ln(embed)  #layernorm is good at the end of blocks
        new_x = x.clone()
        biased_x = torch.cat((new_x, message_x), dim=-1)
        embed_x = self.fc2(biased_x)
        embed_x = message_x
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # reshape att from [batch_size, n_agents, rnn_hidden_dim] to [batch_size * n_agents, rnn_hidden_dim]
        embed_x = embed_x.view(-1, self.args.rnn_hidden_dim)
        h = self.rnn(embed_x, h_in)
        q = self.fc3(h)
        return q, h
