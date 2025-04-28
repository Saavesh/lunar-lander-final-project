# Lunar Lander - Reinforcement Learning Project

---
## Table of Contents
- [Project Description](#project-description)
- [Approach and File Structure](#approach-and-file-structure)
- [Environment Setup](#environment-setup)
- [Result](#result)
- [Conclusion Discussion and Reflection](#conclusion-discussion-and-reflection)
- [References](#references)


## Project Description

This project explores reinforcement learning using the [LunarLander-v3 environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/) from OpenAI Gymnasium. The LunarLander-v3 environment simulates a classic rocket landing task.
The objective is to train or simulate an agent that can land a lunar module safely between two flags, while minimizing fuel usage and maintaining balance.

Rewards are assigned for:
- **Safe landings**
- **Efficient fuel usage**
- **Stable motion**

Penalties are given for:
- **Crashes**
- **Unstable angles**
- **Unnecessary thruster firing**

---


## Approach and Folder Structure
```
lunar-lander-final-project/
├── Lunar-Lander-Final-Project.ipynb        # Main project notebook
├── README.md                               # Project write-up
├── requirements.txt                        # List of dependencies
└── .gitignore                              # Files and folders to ignore
```

---



## Environment Setup

This project was developed using **Python 3.13**.  
Here are the main libraries you’ll need:

| Library | Description | Link |
|:--------|:------------|:-----|
| [Gymnasium](https://gymnasium.farama.org/) | Core RL environments | [GitHub](https://github.com/Farama-Foundation/Gymnasium) |
| [Box2D-py](https://github.com/openai/box2d-py) | Physics engine for 2D simulation (required for LunarLander) | [GitHub](https://github.com/openai/box2d-py) |
| [Matplotlib](https://matplotlib.org/) | Plotting results and visualizations | [Website](https://matplotlib.org/) |
| [Pygame](https://www.pygame.org/) | Rendering game windows (optional) | [Website](https://www.pygame.org/) |
| [NumPy](https://numpy.org/) | Numerical operations | [Website](https://numpy.org/) |


You can install everything  with:

```bash
pip install -r requirements.txt
```

## Result


## Conclusion, Discussion, and Reflection


## References
* [Gymnasium LunarLander-v3 docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
* [PyTorch RL tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [DQN for LunarLander example (yuchen071)](https://github.com/yuchen071/DQN-for-LunarLander-v2)
* [Box2D Physics Engine](https://box2d.org/)


