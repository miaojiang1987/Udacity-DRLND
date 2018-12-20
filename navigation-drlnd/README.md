# Project 1: Navigation

### Introduction

Note: Much of the code/documentation in this repository comes directly from exercises and code provided to me through Udacity's deep reinforcement learning nanodegree program 

This repository provides an agent to learn to collect bananas in a large square world!

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

#### Installation
1. To set up the dependencies, follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)

2. clone this repository

3. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

4. Place the file in this repository and unzip (or decompress) the file. 

#### Training the Agent

##### Command Line
To train the agent, simply run:
```python ./navigation.py```

If you want to watch the agent move through the environment as it learns, run:
```python ./navigation.py --visualize```

For the agent to learn in a reasonable amount of time, it must step through the environment very quickly. When visualizing the agent as it learns, there are two command line arguments to control the speed of the animation how often you want to slow it down. To slow down the animation by 0.1 seconds every 100 episodes, run:
```python ./navigation.py --visualize --slow_by 0.1 --slow_every 100```

The agent will step quickly through the first 99 episodes, and then on the 100th episode, it will slow to a speed where you can watch what it is doing. I found that slow_by values between 0.05 and 0.3 to be the easiest to watch.

##### Jupyter Notebook
There is also a Jupyter notebook saved as Navigation.ipynb if you would prefer to train the agent there.

#### Hyperparameters

All hyperparameters for the agent and the underlying [Deep Q-Networks](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) model are found in dqn_agent.py. Here, you will find options to use [Double DQN](https://arxiv.org/abs/1509.06461), [Dueling DQN](https://arxiv.org/abs/1511.06581), and [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952).

You can also adjust the depth of the network by changing the hyperparameter HIDDEN_LAYERS. This is a list of integers, so to make the network deeper, you simply add another integer to the HIDDEN_LAYERS list. Similarly, the hyperparameter DUELING_LAYERS adjusts the size and number of layers for the dueling streams if you choose to set DUELING_DQN=True.

