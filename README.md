# Reinforcement Learning Collection with PyTorch

This repository is a collection of the following reinforcement learning algorithms:
- **Policy-Gradient**
- **Actor-Critic**
- **Trust Region Policy Optimization**
- **Generalized Advantage Estimation**
- **Proximal Policy Optimization**

More algorithms will be added on this repository.

In this repository, [OpenAI Gym](https://gym.openai.com/) environments such as `CartPole-v0`, `Pendulum-v0`, and `BipedalWalker-v3` are used. You need to install them before running this repository.

*Note*: The environment's names could be different depending on the version of OpenAI Gym.

## Install Dependencies
1. Install Python 3.
2. Install the Python packages in `requirements.txt`. If you are using a virtual environment for Python package management, you can install all python packages needed by using the following bash command:

    ```bash
    $ pip install -r requirements.txt
    ```

3. Install other packages to run OpenAI Gym environments. These are dependent on the development setting of your machine.

4. Install PyTorch. The version of PyTorch should be greater or equal than 1.7.0.

## Training and Running
1. Modify `config.json` as your machine setting.
2. Execute training process by `train.py`. An example of usage for `train.py` are following:

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

![](/assets/img/README/README_2021-01-19-11-04-21.png)

![](/assets/img/README/README_2021-01-19-11-04-28.png)

![](/assets/img/README/README_2021-01-19-11-04-34.png)

![](/assets/img/README/README_2021-01-19-11-04-43.png)

## The results of Pendulum environment

![](/assets/img/README/README_2021-01-19-11-04-50.png)

## The results of BipedalWalker environment

![](/assets/img/README/README_2021-01-19-11-04-58.png)

## Recent Works
- The CUDA usage is provided now.
- Modified some errors in GAE and PPO.
- Modified some errors about horizon was corrected.

## Future Works
- Find the errors of the Actor-Critic
- Implement ACER
- Search other environments to running the algorithms

## References
- An explaination of TRPO line search: [link](https://jonathan-hui.medium.com/rl-trust-region-policy-optimization-trpo-part-2-f51e3b2e373a)
- Additional stability method for PPO value function: [link](https://github.com/takuseno/ppo/issues/6)