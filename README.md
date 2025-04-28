#### Project Description

This project explores reinforcement learning using the [LunarLander-v3 environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/) from OpenAI Gymnasium. The LunarLander-v3 environment simulates a classic rocket landing task.
The objective is to train (or simulate) an agent that can land a lunar module safely between two flags, balancing fuel consumption and landing precision.
Rewards are assigned for safe landings and penalties are given for crashes, fuel waste, and unstable movements.

---


#### 2. Approach
lunar-lander-final-project/
│
├── Lunar-Lander-Final-Project.ipynb    # Project notebook (code)
├── README.md                           # Project writeup
├── run_random_agent.py                 # Script to run random agent
├── images/                             # Folder for screenshots or GIFs
│   ├── lunar_lander_demo.gif           # (image placeholder)
└── rl-env/                             # Virtual environment (ignored by Git)

### Environment Setup
This project was developed in **Python 3.13** with the following main libraries:

| Library | Description | Link |
|:--------|:------------|:-----|
| [Gymnasium](https://gymnasium.farama.org/) | Core RL environments | [GitHub](https://github.com/Farama-Foundation/Gymnasium) |
| [Box2D-py](https://github.com/openai/box2d-py) | Physics engine for 2D simulation (required for LunarLander) | [GitHub](https://github.com/openai/box2d-py) |
| [Matplotlib](https://matplotlib.org/) | Plotting results and visualizations | [Website](https://matplotlib.org/) |
| [Pygame](https://www.pygame.org/) | Rendering game windows (optional) | [Website](https://www.pygame.org/) |

To install all dependencies, you can run:

```bash
pip install gymnasium[box2d] pygame matplotlib

#### Result


#### Conclusion, Discussion, and Reflection


#### Reference
* [Gymnasium LunarLander-v3 docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
* [PyTorch RL tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [DQN for LunarLander example (yuchen071)](https://github.com/yuchen071/DQN-for-LunarLander-v2)






