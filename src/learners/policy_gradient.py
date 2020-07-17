import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.maddpg_critic import MADDPGCritic
from utils.rl_utils import *
import torch as th
from torch.optim import RMSprop


class PGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.target_mac = copy.deepcopy(self.mac)
        self.critic = None
        if args.critic is not None:
            if args.critic == 'maddpg':
                self.critic = MADDPGCritic(scheme, args)
            else:
                raise ValueError("Critic {} not recognised.".format(args.critic))
        self.target_critic = copy.deepcopy(self.critic)

        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
                self.target_mixer = copy.deepcopy(self.mixer)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        if args.mixer is not None:
            if self.mixer == "qmix":
                self.critic_params += list(self.mixer.parameters())

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals, critic_loss, critic_grad_norm, td_error_abs = self._train_critic(batch)

        actions = actions[:,:-1]

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time [batch, time, n_agents, n_actions]

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        if self.args.critic:
            if self.args.critic == 'maddpg':
                q_vals = self.critic(batch, th.argmax(mac_out, dim=-1, keepdim=True))[:, :-1]
                q_vals = q_vals.reshape(-1)
                actor_loss = -(q_vals * mask).sum() / mask.sum()
                # print("We should be here")
            else:
                return NotImplementedError
        else:
            return NotImplementedError


        # Optimise agents
        self.agent_optimiser.zero_grad()
        actor_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if self.args.critic == 'maddpg':
            self._soft_update_targets()
            self.last_target_update_step = t_env

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            if self.args.critic:
                self.logger.log_stat("critic_loss", critic_loss, t_env)
                self.logger.log_stat("td_error_abs", td_error_abs, t_env)
                self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            self.logger.log_stat("q_mean", (q_vals * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (mac_out[:, :-1].reshape(-1, self.n_actions).max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch: EpisodeBatch):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.target_mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick actions taken by each target mac
        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0
        target_act = th.argmax(mac_out, dim=-1, keepdim=True)

        # Calculate y target, we do not need the first time Q-estimation
        target_Q_taken = self.target_critic(batch, target_act)[:, :]  #[batch, time, n_agents, 1]
        target_Q_taken = target_Q_taken.squeeze(-1)  #[batch, time, n_agents]
        if self.args.mixer is not None:
            if self.args.mixer == 'vdn':
                target_Q_taken = target_Q_taken.sum(dim=-1, keepdim=True)
                target_Q_taken = target_Q_taken.repeat(1, 1, self.n_agents)
            elif self.args.mixer == 'qmix':
                target_Q_taken = self.target_mixer(target_Q_taken, batch["state"][:, :])
                target_Q_taken = target_Q_taken.repeat(1, 1, self.n_agents)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

        # 1-step Q or td-lambda
        if self.args.td_lambda:
            y = build_td_lambda_targets(rewards, terminated, mask, target_Q_taken, self.n_agents, self.args.gamma, self.args.td_lambda)
        else:
            # y = rewards + self.args.gamma * (1 - terminated) * target_Q_taken[:, 1:]
            y = build_bootstrap_targets(rewards, terminated, mask, target_Q_taken, self.n_agents, self.args.gamma)

        # Calculate predicted Q
        pred_Q_taken = self.critic(batch)[:, :-1]  # [batch, time, n_agents, 1]
        pred_Q_taken = pred_Q_taken.squeeze(-1) #[batch, time, n_agents]
        if self.args.mixer is not None:
            if self.args.mixer == 'vdn':
                pred_Q_taken = pred_Q_taken.sum(dim=-1, keepdim=True)
                pred_Q_taken = pred_Q_taken.repeat(1, 1, self.n_agents)
            elif self.args.mixer == 'qmix':
                pred_Q_taken = self.mixer(pred_Q_taken, batch["state"][:, :-1])
                pred_Q_taken = pred_Q_taken.repeat(1, 1, self.n_agents)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))

        # Optimise critic
        td_error =  pred_Q_taken - y.detach()
        masked_td_error = td_error * mask

        # normal l2 loss
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()
        self.critic_training_steps += 1
        return pred_Q_taken.detach(), loss.item(), grad_norm, masked_td_error.abs().sum().item()/mask.sum().item()


    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")
        if self.args.mixer:
            if self.args.mixer == 'qmix':
                self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _soft_update_targets(self):
        soft_update(self.target_mac, self.mac, self.args.tau)
        soft_update(self.target_critic, self.critic, self.args.tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
