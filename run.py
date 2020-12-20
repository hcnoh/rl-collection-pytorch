import os
import json
import pickle
import argparse

import numpy as np
import torch
import gym

from models.pg import PolicyGradient
from models.ac import ActorCritic


def main(env_name, model_name, num_episodes, render):
    ckpt_path = ".ckpts/%s/%s/" % (model_name, env_name)

    print(ckpt_path)
    with open(ckpt_path + "model_config.json") as f:
        config = json.load(f)
    
    env = gym.make(env_name)
    env.reset()

    print(env.action_space)

    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]
    
    if model_name == "pg":
        model = PolicyGradient(state_dim, action_dim, discrete, **config)

        model.pi.load_state_dict(torch.load(ckpt_path + "policy.ckpt"))
        if config["train_config"]["use_baseline"]:
            model.v.load_state_dict(torch.load(ckpt_path + "value.ckpt"))
    elif model_name == "ac":
        model = ActorCritic(state_dim, action_dim, discrete, **config)

        model.pi.load_state_dict(torch.load(ckpt_path + "policy.ckpt"))
        model.v.load_state_dict(torch.load(ckpt_path + "value.ckpt"))
    
    for _ in range(num_episodes):
        rwds = []

        done = False
        ob = env.reset()

        while not done:
            act = model.act(ob)
            if render:
                env.render()
            ob, rwd, done, info = env.step(act)
            rwds.append(rwd)
        
        print("The total reward of the episode is %f!" % sum(rwds))
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. The possible environments are [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pg",
        help="Type the model name to train. The possible models are [pg, ac]"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Type the number of episodes to run this agent"
    )
    parser.add_argument(
        "--render",
        type=bool,
        default=True,
        help="Type whether the render is on or not"
    )
    args = parser.parse_args()

    main(args.env_name, args.model_name, args.num_episodes, args.render)