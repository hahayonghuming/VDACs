

# Python Value-Decomposition Actor-Critics (VDACs)

VDACs is a fork of [PyMARL](https://github.com/oxwhirl/pymarl). We added 4 more actor-critics:
- IAC: [Independent Actor-Critics](https://arxiv.org/abs/1705.08926)
- **Naive Critic**: Actor-Critic with a Centralized Critic
- **VDAC-sum**: Proposed Actor-Critic
- **VDAC-mix**: Proposed Actor-Critic

VDACs is written in PyTorch and uses [SMAC](https://github.com/oxwhirl/smac) as its environment.

## Installation instructions
The orginal repository [PyMARL](https://github.com/oxwhirl/pymarl) uses docker to manage the virtual environment. We utilize conda in our implementation:

Create and activate conda virtual environment
```Shell
cd VDACs
conda create --name pymarl python=3.5 
source activate pymarl
git clone git@github.com:hahayonghuming/VDACs.git
cd VDACs
```
Install required packages:
```Shell
pip -r install requirements.txt
```
Set up StarCraft II and SMAC:
```Shell
bash install_sc2.sh
```

## Proposed Algorithms
[Value-decompostion actor-critic](https://arxiv.org/abs/2007.12306) (VDAC) follows an actor-critic approach and is based on three main ideas:
- It is compatible with [A2C](https://arxiv.org/abs/1602.01783), which is proposed to promote RL's training efficiency
- Similar to [QMIX](https://arxiv.org/abs/1803.11485), VDAC enforces the monotonic relationship between global state-value and local state-values <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{V_{tot}}{V^a}&space;\geq&space;0,&space;\forall&space;a&space;\in\{1,\dots,n\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{V_{tot}}{V^a}&space;\geq&space;0,&space;\forall&space;a&space;\in\{1,\dots,n\}" title="\frac{V_{tot}}{V^a} \geq 0, \forall a \in\{1,\dots,n\}" /></a>, which is related to [*difference rewards*](https://pdfs.semanticscholar.org/f5bc/d981ac0cee7e0ba94e738702b30a065ec4d5.pdf)
- VDAC utilizes a simple temporal difference (TD) advantage policy gradient. Both [COMA](https://arxiv.org/abs/1705.08926) advantage gradient and TD advantage gradient are unbiased estimates of a vanilla multi-agent policy gradient. However, our StarCraft testbed results (comparison between **naive critic** and **COMA**) favors TD advantage over COMA advantage

Two VDAC algorithms are proposed:
- **VDAC-sum** simply assumes the global state-value is a summation of local state-values. VDAC-sum does not take advantage of extra state information and shares a similar structure to IAC
![Image of vdnac](https://github.com/hahayonghuming/VDACs/blob/master/train_results/VDN_structure.jpg)
- **VDAC-mix** utilizes a non-negative mixing network as an non-linear function approximator to represent a broader class of functions. The parameters in the mixing network is outputted by a set of hypernetworks, which take input as global states. Therefore, VDAC-mix is capable of incorporating extra state information 
![Image of vmixac](https://github.com/hahayonghuming/VDACs/blob/master/train_results/Vmix.jpg)

## Run the Proposed Algorithms
### Run VDAC-mix

```shell
python3 src/main.py --config=vmix_a2c --env-config=sc2 with env_args.map_name=2s3z
```
### Run VDAC-sum
```shell
python3 src/main.py --config=vdn_a2c --env-config=sc2 with env_args.map_name=2s3z
```
## Run comparison experiments
### Run Naive Critic
```shell
python3 src/main.py --config=central_critic --env-config=sc2 with env_args.map_name=2s3z
```
### Run original QMIX
```shell
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z
```
### Run QMIX with A2C training Paradigm
```shell
python3 src/main.py --config=qmix_a2c --env-config=sc2 with env_args.map_name=2s3z
```
### Run COMA
```shell
python3 src/main.py --config=coma --env-config=sc2 with env_args.map_name=2s3z
```
The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`


All results will be stored in the `Results` folder.

## Training Results
5 independent experiments are conducted for each algorithm on each map. Colored Solid lines denotes the median of the win rate and shades represent the 25-75% percentile. The black dash line represents the win rates of a heurist ai. (My experiments are conducted on a RTX 2080 Ti GPU)

**Note:** We find that **VDAC**s are sensitive to `vf_coef` located in `src/config`. This value penalizes critic losses. In our orignial implementation, we set `vf_coef=0.5`. However, we later find out that `vf_coef=0.1` yields better performance.
### 1c3s5z
![Image of 1c3s5z](https://github.com/hahayonghuming/VDACs/blob/master/train_results/1c3s5zfinal.png)
### 3s5z
![Image of 3s5z](https://github.com/hahayonghuming/VDACs/blob/master/train_results/3s5zfinal.png)
### 2s3z
![Image of 2s3z](https://github.com/hahayonghuming/VDACs/blob/master/train_results/2s3zfinal.png)
### 8m
![Image of 8m](https://github.com/hahayonghuming/VDACs/blob/master/train_results/8mfinal.png)
### bane_vs_bane
![Image of bane_vs_bane](https://github.com/hahayonghuming/VDACs/blob/master/train_results/bane_vs_banefinal.png)
### 2s_vs_1sc
![Image of 2s_vs_1sc](https://github.com/hahayonghuming/VDACs/blob/master/train_results/2s_vs_1scfinal.png)



## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*.
### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. The saved replays can be watched by simply double-clicking on them.

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client. For Windows users who has problem openning replay files, you might need to download a free-trial StarCraft II under the directory ```C:\Program Files (x86)\StarCraft II```

**Description:** Red Units are controlled by **VDAC-mix** and blue ones are controlled by build-in ai which is set to difficulty level 7. Strategies, such as focusing fires on enemies, zealots tend to attack stalkers, can be spotted in replays.
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


## Documentation
This repo is still under development. If you have any questions or concerns, please email js9wv@virginia.edu 

## Citation

If you this repository useful, please cite the following papers:

[VDAC paper](https://arxiv.org/abs/2007.12306).

```tex
@article{su2020value,
  title={Value-Decomposition Multi-Agent Actor-Critics},
  author={Su, Jianyu and Adams, Stephen and Beling, Peter A},
  journal={arXiv preprint arXiv:2007.12306},
  year={2020}
}
```

 [SMAC paper](https://arxiv.org/abs/1902.04043).

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

