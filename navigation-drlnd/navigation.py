import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from time import sleep

from dqn_agent import Agent

from unityagents import UnityEnvironment


def dqn(
  n_episodes=10000
  , max_t=1000
  , eps_start=1.0
  , eps_end=0.01
  , eps_decay=0.995
  , beta_start=0.1
  , beta_end=1.0
  , beta_growth=1.01
  , slow_every=100
  , slow_by=None
  ):
  """Deep Q-Learning.
  
  Params
  ======
    n_episodes (int): maximum number of training episodes
    max_t (int): maximum number of timesteps per episode
    eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    beta_start(float): starting value of beta, for importance-sampling weight in prioritized experience replay
    beta_end (float); maximum value of beta
    beta_growth(float): multiplicative factor (per episode) for increasing beta
    slow_every (int): number of episodes to learn before watching one slowly
    slow_by (float): number of seconds to wait between steps when watching slowly (can be decimal)

  """
  scores = []                        # list containing scores from each episode
  scores_window = deque(maxlen=100)  # last 100 scores
  eps = eps_start                    # initialize epsilon
  beta = beta_start                  # initialize beta
  for i_episode in range(1, n_episodes +1):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    t = 0
    while True:
      action = agent.act(state, eps)                 # select an action
      env_info = env.step(action)[brain_name]        # send the action to the environment
      next_state = env_info.vector_observations[0]   # get the next state
      reward = env_info.rewards[0]                   # get the reward
      done = env_info.local_done[0]                  # see if episode has finished
      agent.step(state, action, reward, next_state, done, beta)
      score += reward                                # update the score
      state = next_state                             # roll over the state to next time step
      if slow_by is not None and i_episode % slow_every == 0:
        sleep(slow_by)
      t += 1
      if done or t > max_t:                                       # exit loop if episode finished
        break
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(eps_end, eps_decay*eps) # decrease epsilon
    beta = min(beta_end, beta_growth*beta) #increase beta
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
      print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window)>=13.0:
      print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
      torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
      return scores
  return scores

def load_agent():
  state_dict = torch.load('checkpoint.pth')
  agent.qnetwork_local.load_state_dict(state_dict)
  agent.qnetwork_target.load_state_dict(state_dict)

def watch_one_episode(slow_by=None):
  env_info = env.reset(train_mode=True)[brain_name] # reset the environment
  state = env_info.vector_observations[0]            # get the current state
  score = 0
  while True:
    action = agent.act(state)                 # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    state = next_state                             # roll over the state to next time step
    score += reward
    print('\rScore: {}'.format(score), end="")
    if slow_by is not None:
      sleep(slow_by)
    if done:                                       # exit loop if episode finished
      break
  


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Solve Banana navigation environment with Q-Networks')
  parser.add_argument('--visualize', dest='visualize', action='store_true', default=False,
                    help='Watch agent learn environment')
  parser.add_argument('--no_train', dest='train', action='store_false', default=True,
                    help='add this if you don\'t want to train the agent')
  parser.add_argument('--load', dest='load', action='store_true', default=False,
                    help='Load a saved model')
  parser.add_argument('--watch_one_episode', dest='watch_one_episode', action='store_true', default=False,
                    help='watch one episode of the agent in action')
  parser.add_argument('--n_episodes', dest='n_episodes', type=int, default=10000,
                    help='max number of episodes to train the agent')
  parser.add_argument('--max_t', dest='max_t', type=int, default=1000,
                    help='max number of timesteps per episode')
  parser.add_argument('--eps_start', dest='eps_start', type=float, default=1.0,
                    help='starting value for epsilon')
  parser.add_argument('--eps_end', dest='eps_end', type=float, default=0.01,
                    help='minimum value for epsilon')
  parser.add_argument('--eps_decay', dest='eps_decay', type=float, default=0.995,
                    help='multiplicative factor (per episode) for decreasing epsilon')
  parser.add_argument('--beta_start', dest='beta_start', type=float, default=1.0,
                    help='starting value for beta')
  parser.add_argument('--beta_end', dest='beta_end', type=float, default=0.01,
                    help='maximum value for beta')
  parser.add_argument('--beta_growth', dest='beta_growth', type=float, default=0.995,
                    help='multiplicative factor (per episode) for increasing beta')
  parser.add_argument('--slow_every', dest='slow_every', type=int, default=100,
                    help='number of episodes before watching the agent slowly')
  parser.add_argument('--slow_by', dest='slow_by', type=float, default=None,
                    help='time to sleep between steps to better visualize environment')

  args = parser.parse_args()

  env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64", no_graphics=not args.visualize and not args.watch_one_episode)

  # get the default brain
  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]

  # reset the environment
  env_info = env.reset(train_mode=True)[brain_name]

  agent = Agent(state_size=len(env_info.vector_observations[0]), action_size=brain.vector_action_space_size, seed=0)

  if args.load:
    load_agent()
  if args.watch_one_episode:
    watch_one_episode(args.slow_by)
  if args.train:
    scores = dqn(
      n_episodes=args.n_episodes
      , max_t=args.max_t
      , eps_start=args.eps_start
      , eps_end=args.eps_end
      , eps_decay=args.eps_decay
      , beta_start=args.beta_start
      , beta_end=args.beta_end
      , beta_growth=args.beta_growth
      , slow_every=args.slow_every
      , slow_by=args.slow_by
    )

  # plot the scores
#  fig = plt.figure()
#  ax = fig.add_subplot(111)
#  plt.plot(np.arange(len(scores)), scores)
#  plt.ylabel('Score')
#  plt.xlabel('Episode #')
#  plt.savefig('scores.png')
#  plt.show()
