<!-- omit from toc -->
# Snake Game using Reinforcement Learning

![animation](./Resources/animation.gif)

**Table of Contents**
- [Project Overview](#project-overview)
  - [Key Features](#key-features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [Game Environment (`game.py`)](#game-environment-gamepy)
  - [Reinforcement Learning Agent (`agent.py`)](#reinforcement-learning-agent-agentpy)
    - [Q-Learning Overview](#q-learning-overview)
  - [Neural Network Model (`model.py`)](#neural-network-model-modelpy)
- [License](#license)



# Project Overview

This project aims to develop an AI agent capable of playing the Snake Game autonomously. It utilizes Reinforcement Learning (RL) techniques, specifically Q-learning, to train the agent to make decisions based on the game's state to maximize its score. The agent learns to navigate the game environment, avoid obstacles, and consume food items to grow in length.

## Key Features

- **Reinforcement Learning Agent**: Implements a Q-learning model to understand and navigate the game environment.
- **Customizable Game Environment**: A Python-based game environment that can be modified for difficulty, size, and appearance.
- **Real-time Training Visualizations**: Offers insights into the agent's learning process through live metrics and performance graphs.

# Installation

## Prerequisites

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib (for visualization)

## Setup

Clone the repository to your local machine:

```
git clone https://github.com/parth-20-07/Snake-Game-using-Reinforcement-Learning.git
cd Snake-Game-using-Reinforcement-Learning
```

Install the required dependencies:

```
pip install -r requirements.txt
```

# Usage

To start the training process, run:

```
python agent.py
```


# How It Works

## Game Environment (`game.py`)

The game environment is designed using Pygame. It creates a grid where the snake moves, controlled either by human input (in `snake_game_human.py`) or by the AI agent (in `agent.py`). The environment generates food items at random positions, and the game's objective is to consume these items to increase the snake's length.

## Reinforcement Learning Agent (`agent.py`)

The agent uses a neural network model (`model.py`) trained with Q-learning, a value-based RL algorithm. The agent observes the state of the environment, selects actions based on its policy, receives rewards based on the outcomes, and updates its policy to improve over time.

### Q-Learning Overview

Q-learning aims to learn a policy that tells an agent what action to take under what circumstances. It does not require a model of the environment and can handle problems with stochastic transitions and rewards without requiring adaptations.

For each state-action pair $(s, a)$, it maintains a Q-value, $Q(s, a)$, representing the expected utility of taking action $a$ in state $s$. The Q-values are updated as:

$$
\begin{equation}
  Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'}Q(s', a') - Q(s, a)]
\end{equation}
$$


where:
- $s$ is the state following state \(s\)
- $r$ is the reward received for moving from \(s\) to \(s'\)
- $\alpha$ is the learning rate
- $\gamma$ is the discount factor

## Neural Network Model (`model.py`)

The neural network model predicts the Q-values for each possible action in a given state. It consists of fully connected layers, with the input layer size depending on the state representation and the output layer size equal to the number of possible actions.


# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
