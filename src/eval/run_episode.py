# 文件：src/eval/run_episode.py

from typing import Tuple, List

import torch

from src.agents.agent import Agent
from src.environment.connect_game_env import ConnectGameEnv


def run_episode(env: ConnectGameEnv,
                agent1: Agent,
                agent2: Agent,
                print_transitions: bool = False,
                initial_actions: List[int] = ()) -> Tuple[dict, List]:
    """
    Runs an entire episode: agent1 versus agent2 in the given environment.
    Returns the episode information and the list of observations (game boards).
    If 'initial_actions' is provided, run those actions before starting the
    episode. Assume that player1 和 player2 按顺序执行 initial_actions（先手是 player1）。

    修改说明：
    1. 先检查 legal 是否为空，如果为空意味着棋盘已满，将其判作平局、结束循环。
    2. 只要 legal 非空，就可以不断“向右偏移”直到找到某个合法列（不用再额外判断“回到原始”）。
    """

    # 重置环境，得到初始观察 obs，和 info
    obs, info = env.reset()
    done = False

    # Run initial_actions，如果这些动作提前结束了游戏就抛错
    for init_action in initial_actions:
        obs, _, done, _ = env.step(action=init_action)
        if done:
            raise ValueError("initial_actions lead to a terminal board")

    # 记录观测序列
    obs_list = [obs]

    if print_transitions:
        print("obs:", obs)

    # 先手是谁？如果 initial_actions 是偶数个，先手就是 agent1；否则先手是 agent2
    active_player = agent1 if len(initial_actions) % 2 == 0 else agent2

    # 游戏主循环
    while not done:
        # 1) 通过 env.active_mark 推断当前玩家编号 0/1
        #    env.active_mark == 1 表示 player1 走 (对应 to_play = 0)
        #    env.active_mark == 2 表示 player2 走 (对应 to_play = 1)
        to_play = 0 if env.active_mark == 1 else 1

        # 2) 生成当前合法动作列表：检查顶格是否为空
        legal = [
            c for c in range(env.ncols)
            if env.board[0, c] == 0
        ]

        # 【改动】如果 legal 为空，说明棋盘已满且无人获胜 —— 平局
        if not legal:
            done = True
            info['is_a_draw'] = True
            break  # 直接退出 while 循环

        # 3) 调用对应 Agent 的 get_action 方法
        with torch.no_grad():
            action = active_player.get_action(
                observation=obs,
                to_play=to_play,
                legal_actions=legal
            )

        # 【改动】只做“不断右移”，直到找到某个合法列
        while action not in legal:
            action = (action + 1) % env.ncols

        # 4) 执行动作，得到下一个 obs、reward、是否结束、info
        obs, reward, done, info = env.step(action=action)

        # 5) 保存新的观测
        obs_list.append(obs)

        # 6) 切换 active_player
        active_player = agent2 if active_player == agent1 else agent1

        if print_transitions:
            print(f"action: {action}, reward: {reward}")
            print("-" * 30)
            print(f"obs: {obs}")
    # 返回本局结束时的 info（包含 winner、rewards1、rewards2、disqualified、is_a_draw 等）和完整的 obs_list
    return info, obs_list


if __name__ == "__main__":
    # DEMO，仅用于测试 run_episode 本身是否能正常工作
    from pprint import pprint
    from src.agents.baselines.random_agent import RandomAgent
    from src.agents.baselines.leftmost_agent import LeftmostAgent
    from src.environment.connect_game_env import ConnectGameEnv

    # 初始化环境
    env = ConnectGameEnv()

    # 初始化 agent1 和 agent2
    agent1 = RandomAgent()
    agent2 = LeftmostAgent()

    # 运行一局并展示结果
    res, obs_list = run_episode(
        env=env,
        agent1=agent1,
        agent2=agent2,
        print_transitions=True,
        initial_actions=[0, 2]
    )
    print()
    pprint(res)
