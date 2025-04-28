# Lunar Lander - Reinforcement Learning Project

---
## Table of Contents
- [Project Description](#project-description)
- [Folder Structure](#folder-structure)
- [Environment Setup](#environment-setup)
- [Result](#result)
- [Conclusion Discussion and Reflection](#conclusion-discussion-and-reflection)
- [References](#references)


## Project Description

This project explores reinforcement learning using the [LunarLander-v3 environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/) from OpenAI Gymnasium. The LunarLander-v3 environment simulates a classic rocket landing task.
The objective is to train (or simulate) an agent that can land a lunar module safely between two flags, balancing fuel consumption and landing precision.
Rewards are assigned for safe landings and penalties are given for crashes, fuel waste, and unstable movements.

---


## Folder Structure
```
lunar-lander-final-project/
├── Lunar-Lander-Final-Project.ipynb
└── README.md
└── .gitignore.md
```




## Environment Setup

This project was developed using **Python 3.13**.  
Here are the main libraries you’ll need:

| Library | Description | Link |
|:--------|:------------|:-----|
| [Gymnasium](https://gymnasium.farama.org/) | Core RL environments | [GitHub](https://github.com/Farama-Foundation/Gymnasium) |
| [Box2D-py](https://github.com/openai/box2d-py) | Physics engine for 2D simulation (required for LunarLander) | [GitHub](https://github.com/openai/box2d-py) |
| [Matplotlib](https://matplotlib.org/) | Plotting results and visualizations | [Website](https://matplotlib.org/) |
| [Pygame](https://www.pygame.org/) | Rendering game windows (optional) | [Website](https://www.pygame.org/) |

To install all dependencies, you can run:

```bash
pip install gymnasium[box2d] pygame matplotlib
```

## Result


## Conclusion, Discussion, and Reflection


## References
* [Gymnasium LunarLander-v3 docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
* [PyTorch RL tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [DQN for LunarLander example (yuchen071)](https://github.com/yuchen071/DQN-for-LunarLander-v2)
* [Box2D Physics Engine](https://box2d.org/)


