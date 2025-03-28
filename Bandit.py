############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


class Visualization():
    def plot1(self, estimated_history, algorithm_name):
        """
        Visualize the performance of each bandit (learning curve) on linear and log scales.
        :param estimated_history: Dictionary mapping bandit ID to list of estimated values over trials.
        :param algorithm_name: Name of the algorithm.
        """
        plt.figure(figsize=(10, 6))
        for arm_id, values in estimated_history.items():
            plt.plot(values, label=f"Bandit {arm_id}")
        plt.xlabel("Trial")
        plt.ylabel("Estimated Value")
        plt.title(f"Learning Curve: {algorithm_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot2(self, cum_rewards_eps, cum_rewards_ts, cum_regrets_eps, cum_regrets_ts):
        """
        Compare cumulative rewards and cumulative regrets for the two algorithms.
        :param cum_rewards_eps: List of cumulative rewards for Epsilon-Greedy.
        :param cum_rewards_ts: List of cumulative rewards for Thompson Sampling.
        :param cum_regrets_eps: List of cumulative regrets for Epsilon-Greedy.
        :param cum_regrets_ts: List of cumulative regrets for Thompson Sampling.
        """
        # cumulative rewards
        plt.figure(figsize=(10, 6))
        plt.plot(cum_rewards_eps, label="Epsilon-Greedy Cumulative Reward")
        plt.plot(cum_rewards_ts, label="Thompson Sampling Cumulative Reward")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()

        # cumulative regrets
        plt.figure(figsize=(10, 6))
        plt.plot(cum_regrets_eps, label="Epsilon-Greedy Cumulative Regret")
        plt.plot(cum_regrets_ts, label="Thompson Sampling Cumulative Regret")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regrets Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()


# A helper class representing an individual advertisement arm.
class AdvertisementArm:
    def __init__(self, true_mean, arm_id):
        """
        :param true_mean: The true reward mean for this advertisement.
        :param arm_id: Identifier for the ad option.
        """
        self.true_mean = true_mean
        self.arm_id = arm_id
        self.n = 0          # number of times this arm was pulled
        self.value = 0.0    # estimated mean reward

    def pull(self):
        """
        Simulate pulling the arm.
        For simplicity, assume a normal reward distribution with standard deviation 1.
        :return: A reward sample.
        """
        return np.random.normal(self.true_mean, 1)

    def update(self, reward):
        """
        Update the estimated reward for this arm.
        Uses an incremental average.
        :param reward: The reward obtained from pulling the arm.
        """
        self.n += 1
        self.value += (reward - self.value) / self.n

    def __repr__(self):
        return f"Arm {self.arm_id}: true_mean={self.true_mean:.2f}, est_value={self.value:.2f}, pulls={self.n}"

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p, trials=20000):
        """
        Initialize the Epsilon-Greedy algorithm.
        :param p: List of true reward means for each advertisement option.
        :param trials: Total number of trials.
        """
        self.trials = trials
        self.arms = [AdvertisementArm(mu, idx) for idx, mu in enumerate(p)]
        self.rewards = []             # Reward at each trial.
        self.selected_arms = []       # Record of selected arm IDs.
        self.cumulative_rewards = []  # Cumulative reward over trials.
        self.cumulative_regret = []   # Cumulative regret over trials.
        self.estimated_history = {arm.arm_id: [] for arm in self.arms}

    def __repr__(self):
        return f"EpsilonGreedy({[str(arm) for arm in self.arms]})"

    def pull(self):
        pass

    def update(self):
        pass

    def experiment(self):
        """
        Epsilon-Greedy experiment.
        Epsilon decays as 1/t.
        :return: Tuple of (total_reward, total_regret)
        """
        total_reward = 0.0
        optimal_mean = max(arm.true_mean for arm in self.arms)
        total_regret = 0.0

        for t in range(1, self.trials + 1):
            epsilon = 1.0 / t  # Decay epsilon by 1/t
            if np.random.rand() < epsilon:
                 chosen_arm = np.random.choice(self.arms)
            else:
                chosen_arm = max(self.arms, key=lambda arm: arm.value)
            self.selected_arms.append(chosen_arm.arm_id)

            reward = chosen_arm.pull()
            chosen_arm.update(reward)
            self.rewards.append(reward)
            total_reward += reward

            # Regret is the difference between the best possible expected reward and the chosen arm's true reward.
            total_regret += (optimal_mean - chosen_arm.true_mean)
            self.cumulative_rewards.append(total_reward)
            self.cumulative_regret.append(total_regret)

            # estimated values for visualization.
            for arm in self.arms:
                self.estimated_history[arm.arm_id].append(arm.value)

            if t % 5000 == 0:
                logger.info(f"EpsilonGreedy trial {t}: cumulative reward = {total_reward:.2f}")

        return total_reward, total_regret

    def report(self):
        """
        Save reward data to CSV and log final cumulative reward and regret.
        """
        df = pd.DataFrame({
            "Bandit": self.selected_arms,
            "Reward": self.rewards,
            "Algorithm": ["Epsilon-Greedy"] * len(self.rewards)
        })
        df.to_csv("epsilon_rewards.csv", index=False)
        logger.info(f"Epsilon-Greedy: Cumulative Reward: {self.cumulative_rewards[-1]:.2f}")
        logger.info(f"Epsilon-Greedy: Cumulative Regret: {self.cumulative_regret[-1]:.2f}")
        return df


class ThompsonSampling(Bandit):
    def __init__(self, p, trials=20000, known_variance=1.0, prior_variance=1.0):
        """
        Initialize the Thompson Sampling algorithm.
        :param p: List of true reward means.
        :param trials: Total number of trials.
        :param known_variance: Assumed known variance of the reward distribution.
        :param prior_variance: Variance of the prior for the mean.
        """
        self.trials = trials
        self.arms = [AdvertisementArm(mu, idx) for idx, mu in enumerate(p)]
        self.known_variance = known_variance
        self.prior_variance = prior_variance
        self.rewards = []
        self.selected_arms = []
        self.cumulative_rewards = []
        self.cumulative_regret = []
        self.estimated_history = {arm.arm_id: [] for arm in self.arms}
        # For posterior estimation
        self.counts = {arm.arm_id: 0 for arm in self.arms}
        self.sum_rewards = {arm.arm_id: 0.0 for arm in self.arms}

    def __repr__(self):
        return f"ThompsonSampling({[str(arm) for arm in self.arms]})"

    def pull(self):
         pass

    def update(self):
         pass

    def experiment(self):
        """
        Thompson Sampling experiment.
        For each arm, if it has been tried, sample from its posterior; otherwise, encourage exploration.
        :return: Tuple of (total_reward, total_regret)
        """
        total_reward = 0.0
        optimal_mean = max(arm.true_mean for arm in self.arms)
        total_regret = 0.0

        for t in range(1, self.trials + 1):
            sampled_means = []
            for arm in self.arms:
                n = self.counts[arm.arm_id]
                if n == 0:
                      sample = np.random.normal(arm.true_mean, np.sqrt(self.prior_variance))
                else:
                    sample_mean = self.sum_rewards[arm.arm_id] / n
                    posterior_std = np.sqrt(self.known_variance / n)
                    sample = np.random.normal(sample_mean, posterior_std)
                sampled_means.append(sample)

            chosen_index = np.argmax(sampled_means)
            chosen_arm = self.arms[chosen_index]
            self.selected_arms.append(chosen_arm.arm_id)

            reward = chosen_arm.pull()
            self.counts[chosen_arm.arm_id] += 1
            self.sum_rewards[chosen_arm.arm_id] += reward

            chosen_arm.update(reward)
            self.rewards.append(reward)
            total_reward += reward
            total_regret += (optimal_mean - chosen_arm.true_mean)
            self.cumulative_rewards.append(total_reward)
            self.cumulative_regret.append(total_regret)

            for arm in self.arms:
                self.estimated_history[arm.arm_id].append(arm.value)

            if t % 5000 == 0:
                logger.info(f"ThompsonSampling trial {t}: cumulative reward = {total_reward:.2f}")

        return total_reward, total_regret

    def report(self):
        """
        Save reward data to CSV and log final cumulative reward and regret.
        """
        df = pd.DataFrame({
            "Bandit": self.selected_arms,
            "Reward": self.rewards,
            "Algorithm": ["Thompson Sampling"] * len(self.rewards)
        })
        df.to_csv("thompson_rewards.csv", index=False)
        logger.info(f"Thompson Sampling: Cumulative Reward: {self.cumulative_rewards[-1]:.2f}")
        logger.info(f"Thompson Sampling: Cumulative Regret: {self.cumulative_regret[-1]:.2f}")
        return df

#--------------------------------------#


def comparison(eps_algo, ts_algo):
    """
    Compare the performance of the two algorithms by visualizing their cumulative rewards and regrets.
    :param eps_algo: Instance of EpsilonGreedy after experiment.
    :param ts_algo: Instance of ThompsonSampling after experiment.
    """
    vis = Visualization()
    vis.plot2(eps_algo.cumulative_rewards, ts_algo.cumulative_rewards,
              eps_algo.cumulative_regret, ts_algo.cumulative_regret)

#--------------------------------------#

# BONUS: A better implementation plan might include separating the experiment runner
# from the algorithm implementations. For example, one could design a generic Experiment class
# that accepts any algorithm (or even multiple runs for statistical significance) and automatically
# tunes hyperparameters or adapts to non-stationary environments (like using sliding windows or discounting).

#--------------------------------------#

if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    # random seed for reproducibility
    np.random.seed(42)

    # True reward means for the four advertisement options
    true_rewards = [1, 2, 3, 4]
    trials = 20000

    eps_algo = EpsilonGreedy(true_rewards, trials)
    logger.info("Starting Epsilon-Greedy Experiment...")
    total_reward_eps, regret_eps = eps_algo.experiment()
    eps_algo.report()

    # Visualization of the learning process for Epsilon-Greedy
    vis = Visualization()
    vis.plot1(eps_algo.estimated_history, "Epsilon-Greedy")

    ts_algo = ThompsonSampling(true_rewards, trials, known_variance=1.0, prior_variance=1.0)
    logger.info("Starting Thompson Sampling Experiment...")
    total_reward_ts, regret_ts = ts_algo.experiment()
    ts_algo.report()

    vis.plot1(ts_algo.estimated_history, "Thompson Sampling")

    comparison(eps_algo, ts_algo)
