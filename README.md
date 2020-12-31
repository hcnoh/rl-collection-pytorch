# Reinforcement Learning Collection with PyTorch

This repository is a collection of reinforcement learning algorithms: Policy-Gradient, Actor-Critic, Trust Region Policy Optimization, and Generalized Advantage Estimation. (More algorithms will be added soon...)

In this repository, [OpenAI Gym](https://gym.openai.com/) environments such as `CartPole-v0`, `Pendulum-v0`, and `BipedalWalker-v3` are used. You need to install them before running this repository.

## Install Dependencies
1. Install Python 3.
2. Install the Python packages in `requirements.txt`. If you are using a virtual environment for Python package management, you can install all python packages needed by using the following bash command:

    ```bash
    $ pip install -r requirements.txt
    ```

3. Install other packages to run OpenAI Gym environments. These are dependent on the development setting of your machine.
4. Install PyTorch. The version of PyTorch should be greater or equal than 1.7.0. This repository does not provide the CUDA usage yet. I will get them possible soon.

## Training and Running
1. Modify `config.json` as your machine setting.
2. Excute training process by `train.py`. An example of usage for `train.py` are following:

    ```bash
    $ python train.py --model_name=trpo --env_name=BipedalWalker-v3
    ```

    The following bash command will help you:

    ```bash
    $ python train.py -h
    ```
3. You can run your pre-trained agents by executing `run.py`. The usage for running `run.py` is similar to that of `train.py`. You can also check the help message by the following bash bash command:

    ```bash
    $ python run.py -h
    ```

## The results of CartPole environment

![](/assets/img/README/README_2020-12-31-11-13-13.png)

![](/assets/img/README/README_2020-12-31-11-13-19.png)

![](/assets/img/README/README_2020-12-31-11-13-27.png)

![](/assets/img/README/README_2020-12-31-11-13-34.png)

## The results of Pendulum environment

![](/assets/img/README/README_2020-12-31-11-13-42.png)

## The results of BipedalWalker environment

![](/assets/img/README/README_2020-12-31-11-13-49.png)

## Future Works
- Find the errors of the Actor-Critic
- Implement PPO
- Search other environments to running the algorithms


### References
- An explaination of TRPO line search: [link](https://jonathan-hui.medium.com/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a)