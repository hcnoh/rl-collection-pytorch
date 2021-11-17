import os
import json
import pickle
import argparse

import torch
import gym

from models.pg import PolicyGradient
from models.ac import ActorCritic
from models.trpo import TRPO
from models.gae import GAE
from models.ppo import PPO


def main(env_name, model_name):
    if not os.path.isdir(".ckpts"):
        os.mkdir(".ckpts")

    if model_name not in ["pg", "ac", "trpo", "gae", "ppo"]:
        print("The model name is wrong!")
        return

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3"]:
        print("The environment name is wrong!")
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

    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if model_name == "pg":
        model = PolicyGradient(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "ac":
        model = ActorCritic(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "trpo":
        model = TRPO(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "gae":
        model = GAE(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "ppo":
        model = PPO(
            state_dim, action_dim, discrete, **config
        ).to(device)

    results = model.train(env)

    env.close()

    with open(ckpt_path + "results.pkl", "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(model.pi.state_dict(), ckpt_path + "policy.ckpt")
    if hasattr(model, "v"):
        torch.save(model.v.state_dict(), ckpt_path + "value.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pg",
        help="Type the model name to train. \
            The possible models are [pg, ac, trpo, gae, ppo]"
    )
    args = parser.parse_args()

    main(**vars(args))
