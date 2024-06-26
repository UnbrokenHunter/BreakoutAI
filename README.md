# Breakout AI using Deep Q-Network (DQN)

This project implements a Deep Q-Network (DQN) to play the Atari game Breakout using PyTorch. The implementation includes an agent, model, environment wrapper, and utilities for training and evaluation.

![Breakout](img/breakout.gif?raw=true "Breakout")

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Agent](#agent)
- [Model](#dueling-network-architecture)
- [Environment](#openai-gym-environment)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/UnbrokenHunter/BreakoutAI.git
   cd BreakoutAI
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Ensure you have gym and torch installed (Also ensure it is a CUDA compatible version):
   ```
   pip install gym torch
   ```

## Usage

#### Training the Model

To train the model, run the following command:

```python
python main.py
```

![Training](img/training_times.png?raw=true "Training")

Don't worry if training takes a while. I found it took ~4,000 epochs before it was even compatent.

#### Testing the Model

To test the model, use:

```python
python test.py
```

## Agent

The `Agent` class in this project is responsible for interacting with the environment, selecting actions, training the model, and evaluating its performance. The agent uses several reinforcement learning concepts and techniques to learn and improve its policy over time.

### Important Concepts

1. **Experience Replay**: By storing and reusing past experiences, the agent learns more efficiently and reduces the correlation between consecutive samples.
2. **Double DQN**: This method uses two separate networks (online and target) to mitigate overestimation bias in Q-learning. The online network is updated frequently, while the target network is updated less often, providing stable targets for the Q-learning updates.
3. **Epsilon-Greedy Policy**: This policy encourages exploration by selecting random actions with a certain probability (epsilon) and the best-known action otherwise. As training progresses, epsilon is gradually reduced to favor exploitation.
4. **Target Network**: The target network is a slowly updated copy of the online network. This stabilization technique helps in providing consistent Q-value targets, improving the convergence of the learning algorithm.
5. **Batch Training**: The agent samples mini-batches from the replay memory to update the network. This method improves learning stability and efficiency by using a diverse set of experiences for each update.

These components work together to enable the agent to learn effective policies for playing Breakout, achieving high scores through improved decision-making and policy evaluation.

## Dueling Network Architecture

This project incorporates a dueling network architecture for deep reinforcement learning, as described in the paper "Dueling Network Architectures for Deep Reinforcement Learning" by Ziyu Wang et al. The key idea behind this architecture is to separate the representation of state values and state-dependent action advantages.

![Model](img/DQN.png?raw=true "Model")

### Key Features

1. **Two-Stream Architecture**: The network is split into two streams, one for the value function and one for the advantage function, sharing a common convolutional feature learning module.
2. **Aggregation Layer**: These streams are combined using an aggregation layer to produce the Q-values. This approach helps in better policy evaluation, especially in the presence of many similar-valued actions.
3. **State-Value Function**: The value stream estimates how good it is to be in a particular state, which is crucial for temporal-difference-based methods like Q-learning.
4. **Action Advantage Function**: The advantage stream estimates the relative importance of each action in a given state, which helps in selecting the best action to take.

### Benefits

- **Better Policy Evaluation**: By separating value and advantage, the network can more quickly identify the correct action, leading to improved performance, especially in environments with many actions.
- **Improved Stability**: The dueling architecture provides more stable updates as it separates the estimation of state values from action advantages.

### Reference

For more info, refer to the paper:

- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

## OpenAI Gym Environment

OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. It provides a wide variety of environments ranging from classic control problems to Atari games, allowing for standardized testing and benchmarking of algorithms.

### Breakout Environment

In this project, we use the Breakout environment from the Atari suite of games provided by OpenAI Gym. Breakout is a classic arcade game where the player controls a paddle to hit a ball and break bricks at the top of the screen. The objective is to break all the bricks while preventing the ball from falling off the bottom of the screen.

### Key Features

1. **State Representation**: The state is represented as a sequence of frames (images) showing the current screen of the game.
2. **Action Space**: The action space consists of discrete actions, such as moving the paddle left, right, or not moving.
3. **Rewards**: The agent receives rewards for breaking bricks and loses the game if the ball falls off the screen.

### Preprocessing

To enhance learning efficiency, the environment is wrapped with custom preprocessing steps:

- **Frame Skipping and Stacking**: To reduce the amount of computation and to provide temporal context, we skip and stack frames.
- **Grayscale Conversion**: The frames are converted to grayscale to reduce the state representation size.
- **Resizing**: The frames are resized to a smaller resolution suitable for the neural network input.

### Usage Example

The Breakout environment is wrapped in the `DQNBreakout` class which handles all the necessary preprocessing:

```python
from breakout import DQNBreakout

# Initialize the Breakout environment
environment = DQNBreakout(device=device)
```

By using this environment, the agent can interact with the game, receive rewards, and learn to play Breakout through reinforcement learning algorithms.

For more info, refer to the Documentation:

- [OpenAI Gym Documentation](https://www.gymlibrary.dev/environments/atari/breakout/)
