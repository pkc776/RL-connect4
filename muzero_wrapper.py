# 文件路径：src/agents/muzero_wrapper.py

import numpy as np
from typing import List, Tuple

# 导入你的新环境 ConnectGameEnv
# 请根据实际路径修改下面这一行
# 假设 ConnectGameEnv 在 src/environment/connect_game_env.py：
from src.environment.connect_game_env import ConnectGameEnv


class MuZeroStateWrapper:
    """
    这个包装器不再自己维护一套环境，而是“绑定”在一个已有的 ConnectGameEnv 实例上，
    把它的内部 board、active_mark 转成 MuZero 期待的 (3, 6, 7) 三通道观测，以及
    提供 to_play(), legal_actions() 等接口，使 MuZeroAgent 可以在同一个核心环境上运行。
    """

    def __init__(self, core_env: ConnectGameEnv):
        """
        :param core_env: 已经 new 出来的 ConnectGameEnv 实例（比赛时由外部 new）
        """
        self.env = core_env
        # 记录行列数
        self.nrows = self.env.nrows
        self.ncols = self.env.ncols

    def get_obs(self) -> np.ndarray:
        """
        MuZero 期待的观测格式是 (3, nrows, ncols)，dtype float32：
          通道0：board == 1 (player1) 的位置设为 1，其余设 0
          通道1：board == 2 (player2) 的位置设为 1，其余设 0
          通道2：全盘填充“当前行动者”对应符号：player1 -> 1.0，player2 -> -1.0
        这里 env.active_mark 是 1 或 2
        """
        board = self.env.board              # 绝对棋盘，shape=(6,7)，值 ∈ {0,1,2}
        active_mark = self.env.active_mark  # 1 或 2

        board_player1 = (board == 1).astype(np.float32)  # shape=(6,7)
        board_player2 = (board == 2).astype(np.float32)  # shape=(6,7)
        mark_val = 1.0 if active_mark == 1 else -1.0
        board_to_play = np.full((self.nrows, self.ncols), mark_val, dtype=np.float32)

        obs = np.stack([board_player1, board_player2, board_to_play], axis=0)
        return obs  # dtype float32, shape=(3,6,7)

    def to_play(self) -> int:
        """
        MuZero 旧版 players=[0,1]，0 代表 player1，1 代表 player2。
        env.active_mark == 1 -> 返回 0；env.active_mark == 2 -> 返回 1
        """
        return 0 if self.env.active_mark == 1 else 1

    def legal_actions(self) -> List[int]:
        """
        ConnectGameEnv 本身并没有直接给出 legal_actions()，但我们知道
        如果某列顶部 (board[nrows-1][col]) == 0 就代表该列可以下
        """
        board = self.env.board
        legal = []
        for col in range(self.ncols):
            if board[self.nrows - 1][col] == 0:
                legal.append(col)
        return legal

    ## —— 以下接口只是为了 MuZeroAgent 在测试/人机对战时使用 —— ##
    def render(self) -> None:
        """
        把棋盘打印到控制台，暂停等待用户回车
        """
        board = self.env.board
        lines = []
        for row in board[::-1]:
            line = []
            for col in row:
                if col == 1:
                    ch = "o"
                elif col == 2:
                    ch = "x"
                else:
                    ch = " "
                line.append(ch)
            lines.append(" ".join(line))
        lines.append("-" * (2 * self.ncols))
        lines.append(" ".join(str(i + 1) for i in range(self.ncols)))
        print("\n".join(lines))
        input("Press enter to continue")

    def human_to_action(self) -> int:
        """
        MuZero 在 human vs AI 模式下调用，让用户输入一个合法列 (0~ncols-1)
        """
        to_play = self.to_play()
        prompt = f"Enter the column to play for player {to_play} (0-{self.ncols-1}): "
        choice = input(prompt)
        while (not choice.isnumeric()) or (int(choice) not in self.legal_actions()):
            choice = input(f"Enter a valid column (0-{self.ncols-1}): ")
        return int(choice)

    def expert_agent(self) -> int:
        """
        MuZero config.opponent="expert" 时调用。这里暂时返回一个随机合法动作
        """
        legal = self.legal_actions()
        return np.random.choice(legal) if len(legal) > 0 else 0

    def action_to_string(self, action_number: int) -> str:
        """
        MuZero 在 human vs AI 模式时展示动作
        """
        return f"Play column {action_number + 1}"
