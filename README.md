# Lunar Lander - Reinforcement Learning Project

In this project, we train a reinforcement learning agent to land a spacecraft safely in the LunarLander-v3 environment using a Deep Q-Network (DQN).

This project explores reinforcement learning using the [LunarLander-v3 environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/) from OpenAI Gymnasium. The environment simulates a classic rocket landing task, where the goal is to land a lunar module safely between two flags while minimizing fuel usage and keeping the craft balanced.


---

## Project Files
- lunar_lander_dqn_training.ipynb — Main Jupyter notebook with full training and explanations.
- training_curve.png — Graph showing agent’s training progress over episodes.
- dqn_lander.pth — Saved model weights for the trained agent.
- requirements.txt — Python libraries needed to run the project.
- README.md — Project overview and instructions.


---



## How to Run

This project was developed using **Python 3.13**.  
1. Install Dependencies
```bash
pip install -r requirements.txt
```

2. Launch Jupyter notebook
 ``` bash
jupyter notebook
```

3. Open lunar_lander_dqn_training.ipynb and run the cells to train or evaluate the agent.

---


## References
[1] [Gymnasium LunarLander-v3 Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

[2] [PyTorch Reinforcement Learning Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

[3] [Box2D Physics Engine](https://box2d.org/)

[4] [Example Project: DQN for LunarLander-v2 by yuchen071 (GitHub)](https://github.com/yuchen071/DQN-for-LunarLander-v2)


