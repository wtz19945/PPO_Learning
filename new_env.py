import gym
import numpy as np

MOTORS_TORQUE = 80

class CustomBipedalWalker(gym.Env):
    def __init__(self):
        self.env = gym.make('BipedalWalker-v3')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = self.calculate_reward(observation, reward, done, info, action)
        return observation, reward, done, info

    def calculate_reward(self, observation, reward, done, info, action):
        # Custom reward calculation based on observation, reward, done, info
        # Your reward function logic goes here
        reward = reward
        #

        # Penalty of torque used
        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less
        # Large penalty on failure
        if done:
            reward = -100

        return reward

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

# Example usage
if __name__ == '__main__':
    env = CustomBipedalWalker()
    observation = env.reset()
    while True:
        action = env.action_space.sample()  # Replace with your agent's action
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
