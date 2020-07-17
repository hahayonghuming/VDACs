from envs import REGISTRY as env_REGISTRY
from tqdm import tqdm, trange
from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import pprint
import numpy as np

def heuristic_run(n_episodes, map_name, env_args):
    env_args['map_name'] = map_name
    pprint.pprint(env_args)
    env = StarCraft2Env(**env_args)
    wins = 0
    with trange(n_episodes) as t:
        for i in t:
            env.reset()
            terminated = False
            episode_reward = 0

            while not terminated:
                actions = []
                for agent_id in range(env.n_agents):
                    avail_actions = env.get_avail_agent_actions(agent_id)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    action = np.random.choice(avail_actions_ind)
                    # _, haction_num = env.get_agent_action_heuristic(agent_id, action)
                    actions.append(action)

                reward, terminated, info = env.step(actions)
            try:
                wins += info['battle_won']
            except:
                continue
            t.set_postfix(win_rate=wins/(i+1.))
        env.close()
    print("\n")
    print("In {} episodes games, heuristic ai wins {}; win rate is {}".format(n_episodes, wins, float(wins)/float(n_episodes)))


if __name__ == '__main__':
    import os
    import yaml
    config_name = 'sc2'
    with open(os.path.join(os.path.dirname(__file__), "config", "envs", "{}.yaml".format(config_name)), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(config_name, exc)
    env_args = config_dict['env_args']
    env_args['heuristic_ai'] = True
    map_name = '2s_vs_1sc'
    heuristic_run(1000, map_name, env_args)
