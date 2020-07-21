

# Python Value-Decomposition Actor-Critics (VDACs)

VDACs is a fork of [PyMARL](https://github.com/oxwhirl/pymarl/tree/master/src). We added 4 more actor-critics:
- IAC: [Independent Actor-Critics](https://arxiv.org/abs/1705.08926)
- Naive Critic: Actor-Critic with a Centralized Critic
- VDAC-sum: Proposed Actor-Critic
- VDAC-mix: Proposed Actor-Critic

VDACs is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions

Build the Dockerfile
```Shell
cd docker
bash build.sh
```
Set up StarCraft II and SMAC:
```Shell
bash install_sc2.sh
```
Optionaly, you can use pip to install required packages using the requirement.txt file if you have troubling using docker
## Proposed Algorithms
Value-decompostion actor-critic (VDAC) is based on three main ideas:
- It is compatible with [A2C](https://arxiv.org/abs/1602.01783), which is proposed to promote RL's training efficiency
- Similar to [QMIX](https://arxiv.org/abs/1803.11485), VDAC enforces the monotonic relationship between global state-value and local state values $$\frac{V_{tot}}{V^a}\geq 0, \forall a \in {0,\dots,n}$$, which is related to [difference rewards](https://pdfs.semanticscholar.org/f5bc/d981ac0cee7e0ba94e738702b30a065ec4d5.pdf)
- VDAC utilize a simple temporal difference (TD) advantage policy gradient. Both [COMA](https://arxiv.org/abs/1705.08926) advantage gradient and temporal difference advantage gradient are unbiased estimates of a vanilla policy gradient. However, our StarCraft testbed results (comparison between naive critic and COMA) favors TD advantage over COMA advantage.
## Run the Proposed Algorithms

```shell
python3 src/main.py --config=vmix_a2c --env-config=sc2 with env_args.map_name=2s3z
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To run experiments using the Docker container:
```shell
bash run.sh $GPU python3 src/main.py --config=vmix_a2c --env-config=sc2 with env_args.map_name=2s3z
```

All results will be stored in the `Results` folder.

## Training Results
5 independent experiments are conducted for each algorithm on each map. Colored Solid lines denotes the median of the win rate and shades represent the 25-75% percentile. The black dash line represents the win rates of a heurist ai. 
### 1c3s5z
![Image of 1c3s5z](https://github.com/hahayonghuming/VDACs/blob/master/train_results/1c3s5z(1).png)
### 3s5z
![Image of 3s5z](https://github.com/hahayonghuming/VDACs/blob/master/train_results/3s5z(1).png)
### 2s3z
![Image of 2s3z](https://github.com/hahayonghuming/VDACs/blob/master/train_results/2s3z(1).png)
### 8m
![Image of 8m](https://github.com/hahayonghuming/VDACs/blob/master/train_results/8m(1).png)
### bane_vs_bane
![Image of bane_vs_bane](https://github.com/hahayonghuming/VDACs/blob/master/train_results/bane_vs_bane(1).png)
### 2s_vs_1sc
![Image of 2s_vs_1sc](https://github.com/hahayonghuming/VDACs/blob/master/train_results/2s_vs_1sc(1).png)



## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*.
### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. The saved replays can be watched by simply double-clicking on them.

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client. For Windows users who has problem openning replay files, you might need to download a free-trial StarCraft II under the directory ```C:\Program Files (x86)\StarCraft II```
### 1c3s5z
![Image of 1c3s5z](https://github.com/hahayonghuming/VDACs/blob/master/replays/1c3s5z.gif)
### 3s5z
![Image of 3s5z](https://github.com/hahayonghuming/VDACs/blob/master/replays/3s5z.gif)
### 2s3z
![Image of 2s3z](https://github.com/hahayonghuming/VDACs/blob/master/replays/2s3z.gif)
### 8m
![Image of 8m](https://github.com/hahayonghuming/VDACs/blob/master/replays/8m.gif)
### bane_vs_bane
![Image of bane_vs_bane](https://github.com/hahayonghuming/VDACs/blob/master/replays/bane_vs_bane.gif)




## Citation

If you this repository useful, please cite the following papers:
[SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

 [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

## License

Code licensed under the Apache License v2.0
