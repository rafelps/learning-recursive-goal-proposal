# Learning Recursive Goal Proposal: A Hierarchical Reinforcement Learning approach (LRGP)

Code for my final master thesis [Learning Recursive Goal Proposal: A Hierarchical Reinforcement Learning approach]
[thesis].

This master thesis presents [Learning Recursive Goal Proposal (LRGP)][thesis], a new hierarchical algorithm based on two
levels in which the higher one serves as a goal proposal for the lower one, which interacts with the environment
following the proposed goals. The main idea of this novel method is to break a task into two parts, speeding and easing
the learning process. In addition to this, [LRGP][thesis] implements a new reward system that takes advantage of
non-sparse rewards to increase its sample efficiency by generating more transitions per episode, which are stored and
reused thanks to Experience Replay. [LRGP][thesis], which has the flexibility to be used with a wide variety of
Reinforcement Learning algorithms in environments of different nature, obtains State-of-the-Art results both in
performance and efficiency when compared to methods such as [Double DQN][ddqn] or [Soft Actor Critic (SAC)][sac] in
[Simple MiniGrid][smg] and [Pendulum][pend] environments.

## Dependencies

This project has been developed using:

- [Python][python] 3.7
- [PyTorch][pytorch] 1.7.1
- [NumPy][numpy] 1.19.2
- [OpenAI Gym][gym] 0.17.2
- [Gym Simple MiniGrid][smg] 2.0.0

## Usage

This repository contains two main scripts: `train_lrgp.py` and `test_lrgp.py`.

### `train_lrgp.py`

This script serves to train our model, log the process' metrics and export a checkpoint. The complete usage of the
script can be seen using the help option, which presents a self-explanatory message about the diferent parameters:

```
$ python train_lrgp.py -h
```

Apart from common Reinforcement Learning arguments, this script requires `--job_name JOB_NAME` to identify the training
process and create a directory `logs/JOB_NAME` in which the final checkpoint and the training logs will be stored.

The [Thesis][thesis] provides a detailed explanation for each of the script parameters, both for basic RL
arguments&mdash;such as `n_episodes` or `test_each`&mdash;and for method-specific ones like `low_h` which defines the
maximun number of steps the low agent can take each run; or `high_h`, which controls the maximum number of goal
proposals available for each episode.

Additionally, there are many other parameters that have not been included in the main script arguments as we use default
values on them for all experiments. An example can be the learning rate for the different learning algorithms, or the
number of hidden layers for the neural networks. These can be easily modified in the respective files or even included
in the main arguments' list.

*Update*: The epsilon parameter, which controls the amount of exploration of the algorithm (following an epsilon-greedy
policy) is now defined using an exponentially-decayed function, while in the original publication it was defined 
using a piece-wise function. The parameters have been set to obtain the most similar performance.

### `test_lrgp.py`

The goal of this script is to test an already-trained checkpoint, obtain performance metrics and optionally 
visualize the learned policy.

The complete usage of the script can be seen using:

```
$ python test_lrgp.py -h
```

Its main arguments are:

- `--checkpoint_name CHECKPOINT_NAME`: Name that was given to the training job to be tested. The script will 
  automatically look for the checkpoint at the path `checkpoints/CHECKPOINT_NAME`, which should have the same 
  internal structure and filenames than the logging directory generated by the training script.
  
- `--render`: Flag to visualize the learned policy.

*Note*: The checkpoint uploaded in this repository has been obtained using this version of the algorithm and shows 
slightly different metrics than the ones reported in the [Thesis][thesis] due to the changes in the epsilon 
parameter.

## Learned Policy
<p align="center">
  <img src="figures/SimpleFourRoomsEnv15x15.gif" width="360" alt="Simple-MiniGrid-FourRooms-15x15-v0 learned policy">
</p>

## Citation

```bibtex
@phdthesis{Palliser Sans_2021,
    title={Learning recursive goal proposal: a hierarchical reinforcement learning approach},
    url={http://hdl.handle.net/2117/348422},
    abstractNote={Reinforcement Learning's unique way of learning has led to remarkable successes like Alpha Zero or 
        Alpha Go, mastering the games of Chess and Go, and being able to beat the respective World Champions. 
        Notwithstanding, Reinforcement Learning algorithms are underused in real-world applications compared to other 
        techniques such as Supervised or Unsupervised learning. One of the most significant problems that could limit 
        the applicability of Reinforcement Learning in other areas is its sample inefficiency, i.e., its need for vast 
        amounts of interactions to obtain good behaviors. Off-policy algorithms, those that can learn a different policy 
        than the one they use for exploration, take advantage of Replay Buffers to store the gathered knowledge and 
        reuse it, already making a step into sample efficiency. However, in complex tasks, they still need lots of 
        interactions with the environment to explore and learn good policies. This master thesis presents Learning 
        Recursive Goal Proposal (LRGP), a new hierarchical algorithm based on two levels in which the higher one serves 
        as a goal proposal for the lower one, which interacts with the environment following the proposed goals. The 
        main idea of this novel method is to break a task into two parts, speeding and easing the learning process. In 
        addition to this, LRGP implements a new reward system that takes advantage of non-sparse rewards to increase its 
        sample efficiency by generating more transitions per episode, which are stored and reused thanks to Experience 
        Replay. LRGP, which has the flexibility to be used with a wide variety of Reinforcement Learning algorithms in 
        environments of different nature, obtains State-of-the-Art results both in performance and efficiency when 
        compared to methods such as Double DQN or Soft Actor Critic (SAC) in Simple MiniGrid and Pendulum environments.},
    school={UPC, Facultat d'Informàtica de Barcelona, Departament de Ciències de la Computació},
    author={Palliser Sans, Rafel},
    year={2021},
    month={Apr}
}
```

[thesis]: http://hdl.handle.net/2117/348422

[ddqn]: https://arxiv.org/abs/1509.06461

[sac]: https://arxiv.org/abs/1801.01290

[smg]: https://github.com/rafelps/gym-simple-minigrid

[pend]: https://gym.openai.com/envs/Pendulum-v0/

[python]: https://www.python.org/

[pytorch]: https://pytorch.org/

[numpy]: https://numpy.org/

[gym]: https://gym.openai.com/
