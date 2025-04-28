#!/usr/bin/env python3
"""
run_random_agent.py

Quick sanity‐check: we’ll tumble the lander around randomly for a set number of steps 
so you can make sure Gymnasium, Box2D, and pygame are all talking to each other.
"""

import argparse
import gymnasium as gym

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a totally random agent on any Gymnasium environment."
    )
    p.add_argument(
        "--env",
        type=str,
        default="LunarLander-v3",
        help="Which Gym ID to load (default: LunarLander-v3)"
    )
    p.add_argument(
        "--steps", type=int, default=1000,
        help="How many steps to run before quitting (default: 1000)"
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for repeatability (default: no seed)"
    )
    p.add_argument(
        "--render", choices=["human", "rgb_array", "ansi", None],
        default="human",
        help="Rendering mode (default: pop up a window)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # create the env and immediately pop up the window if you're in human mode
    env = gym.make(args.env, render_mode=args.render)

    # start things off (and maybe seeds for reproducibility)
    obs, info = env.reset(seed=args.seed)

    for i in range(1, args.steps + 1):
        # pick a random action just to see the physics in action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            # when an episode finishes, restart cleanly
            obs, info = env.reset(seed=args.seed)

    env.close()
    print(f"All done! Ran {args.steps} steps on {args.env}.")

if __name__ == "__main__":
    main()