import numpy as np
from torch.distributions import Categorical

from src.agents.baselines.baseline_agent import BaselineAgent


class RandomAgent(BaselineAgent):
    """
    RandomAgent: BaselineAgent that selects one of the columns at random
    """

    def __init__(self, name: str = "Random Agent", **kwargs) -> None:
        super(RandomAgent, self).__init__(name=name, **kwargs)

    def get_exploitation_policy(self, obs: np.ndarray) -> Categorical:
        """
        Returns the policy to follow in order to exploit the environment
        in the given observation. The RandomAgent only knows exploration.

        :param obs: environment observation (game board)
        :return: distribution over the action space to exploit the environment
        """

        policy = self.get_exploration_policy(obs=obs)
        return policy

    # 把 obs 改成 observation，並加上 **kwargs
    def get_action(self, observation, legal_actions, **kwargs):
        """
        傳入：
          observation    -- 當前棋局的觀察 (game board)，這裡可以忽略
          legal_actions  -- 一個可下 column 的列表 (例如 [0, 1, 3, 5])
          **kwargs       -- 其他多餘的 keyword（例如 player_id、info 等），先接住以免出錯
        回傳：
          在 legal_actions 中隨機選一個 column index，作為下一步。
        """
        # legal_actions 可能是 list 或 numpy array，先確保它是 list
        la = list(legal_actions)
        # 隨機從合法動作裡面挑一個
        choice = np.random.choice(la)
        return choice


if __name__ == "__main__":
    # DEMO
    from src.environment.connect_game_env import ConnectGameEnv

    agent = RandomAgent()
    print(agent.name)
    obs = ConnectGameEnv.random_observation()
    print('obs =\n', obs)
    transition = agent.get_transition(state=obs)
    policy = agent.get_exploitation_policy(obs=obs)
    print('policy =\n', policy.probs)
    vis_policy = agent.get_policy_scores_to_visualize(obs=obs)
    print('visualization policy =\n', vis_policy)
    print('action =', transition['action'])
    print('\ntransition =\n', transition)
    print('\nsymmetric transition=\n', agent.get_symmetric_transition(transition))
