# Lunar Lander - Reinforcement Learning Project

In this project, we train a reinforcement learning agent to land a spacecraft safely in the LunarLander-v3 environment using a Deep Q-Network (DQN).

We explore reinforcement learning using the 
[LunarLander-v3 environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/) from OpenAI Gymnasium. The environment simulates a classic rocket landing task, where the goal is to land a lunar module safely between two flags while minimizing fuel usage and keeping the craft stable.

This project experiments with different model configurations to improve performance, including adjusting network size and learning rates.



---

## Project Files
- `lunar_lander_dqn_training.ipynb` — Main notebook with full model building, training, testing, results, and demo clip.
- `images/` — Folder containing saved training curves for each model run.
- `demo_lander.gif` — Demo animation of the trained agent successfully landing.
- `dqn_lander.pth`, `dqn_lander_run2.pth`, `dqn_lander_run3.pth` — Saved trained models.
- `requirements.txt` — List of required Python packages.

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

3. Open lunar_lander_dqn_training.ipynb
   - You can retrain the agent or evaluate the pre-trained models.
   - Training graphs and results are automatically generated during execution.
   - A demo GIF (demo_lander.gif) of the trained lander is already included for quick viewing.

---


## References

[1] [Gymnasium GitHub Repository](https://github.com/Farama-Foundation/Gymnasium)

[2] [Gymnasium LunarLander-v3 Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

[3] [Deep Reinforcement Learning Resources - Gymnasium Farama Docs](https://gymnasium.farama.org/tutorials/)

[4] [PyTorch Reinforcement Learning Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

[5] [Box2D Physics Engine](https://box2d.org/)

[6] [Example Project: DQN for LunarLander-v2 by yuchen071 (GitHub)](https://github.com/yuchen071/DQN-for-LunarLander-v2)

[7] [Original Lunar Lander Reinforcement Learning Guide by sokistar24 (GitHub)](https://github.com/sokistar24/Deep_Reinforcement_learning)

[8] [Deep Q-Learning Paper (Mnih et al., 2015)](https://arxiv.org/abs/1312.5602)

[9] [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

[10] [Double DQN: Deep Reinforcement Learning with Double Q-learning (Van Hasselt et al., 2016)](https://arxiv.org/abs/1509.06461)

[11] [Dueling Network Architectures for Deep Reinforcement Learning (Wang et al., 2016)](https://arxiv.org/abs/1511.06581)

[12] [Prioritized Experience Replay (Schaul et al., 2015)](https://arxiv.org/abs/1511.05952)