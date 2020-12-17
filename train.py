import os
import json
import pickle
import argparse

import numpy as np
import torch
import gym

from models.pg import PolicyGradient
from models.ac import ActorCritic


def main(env_name, model_name):
    if not os.path.isdir(".ckpts"):
        os.mkdir(".ckpts")
    
    if model_name not in ["pg", "ac"]:
        print("The model name is wrong!")
        return
    
    ckpt_path = ".ckpts/%s/" % model_name
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    ckpt_path = ckpt_path + "%s/" % env_name
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name][model_name]
    
    with open(ckpt_path + "model_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
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
    # action_high = np.max(env.action_space.high)
    # action_low = np.min(env.action_space.low)
    # action_range = np.maximum(np.abs(action_high), np.abs(action_low))

    cuda = torch.cuda.is_available()

    if model_name == "pg":
        model = PolicyGradient(state_dim, action_dim, discrete, **config)
    elif model_name == "ac":
        model = ActorCritic(state_dim, action_dim, discrete, **config)
    
    if cuda:
        for net in model.get_networks():
            net = net.cuda()

    results = model.train(env)
    
    env.close()

    with open(ckpt_path + "results.pkl", "wb") as f:
        pickle.dump(results, f)

    if model_name == "pg":
        torch.save(model.pi.state_dict(), ckpt_path + "policy.ckpt")
        if config["train_config"]["use_baseline"]:
            torch.save(model.v.state_dict(), ckpt_path + "value.ckpt")
    elif model_name == "ac":
        torch.save(model.pi.state_dict(), ckpt_path + "policy.ckpt")
        torch.save(model.v.state_dict(), ckpt_path + "value.ckpt")


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
    args = parser.parse_args()

    main(args.env_name, args.model_name)