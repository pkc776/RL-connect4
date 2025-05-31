# wrapped_connect4.py

import numpy as np

# 这里的导入路径请根据你项目实际存放 ConnectGameEnv 的位置来改：
# 假设你把 "新环境" 的代码保存在 src/environment/connect_game_env.py
# 那么这行就写成：
#   from src.environment.connect_game_env import ConnectGameEnv
#
# 如果你直接把新环境代码放在项目根目录下的 connect_env.py，则写成：
#   from connect_env import ConnectGameEnv
#
# 下面示例假设它在 src/environment/connect_game_env.py：
from src.environment.connect_game_env import ConnectGameEnv  
from typing import Tuple

class WrappedConnect4:
    """
    这个类把“新环境” ConnectGameEnv 包装成 MuZero 旧接口 (3,6,7) 的格式：
      - reset() 返回 obs: np.ndarray, shape=(3,6,7), dtype=float32
      - step(action) 返回 (obs, reward, done)
      - to_play() 返回 0 或 1
      - legal_actions() 返回合法动作列表
      - render(), human_to_action() 则仿照旧环境格式
    """

    def __init__(self, nrows: int = 6, ncols: int = 7, inrow: int = 4):
        # 1. 用新的游戏环境构造器初始化
        #    这里我们把 nrows=6, ncols=7, inrow=4 作为默认值，
        #    你也可以根据需要在 MuZero Game 类里改成别的。
        self.env = ConnectGameEnv(nrows=nrows, ncols=ncols, inrow=inrow)

        # 缓存一下行列数，方便后面拼 shape
        self.nrows = nrows
        self.ncols = ncols

    def reset(self) -> np.ndarray:
        """
        启动一局新游戏，返回 MuZero 期待的三通道观测 (3, 6, 7)：
          - 第 0 通道：所有 board==1 的位置（player1）置 1、其余 0
          - 第 1 通道：所有 board==2 的位置（player2）置 1、其余 0
          - 第 2 通道：整个板子填充当前行动者 (player) 对应的符号：1 或 -1
        """
        # 调用新环境的 reset 方法。它会返回 (obs_relative, info)，
        # 但我们真正需要的是 env.board（内部绝对棋盘）和 env.active_mark（当前行动者）。
        _obs_rel, _info = self.env.reset(init_random_obs=False)

        # 此时内部：
        #   self.env.board       : np.ndarray, shape=(6,7), 值 {0,1,2}，其中 1=player1, 2=player2
        #   self.env.active_mark : 1 或 2，表示下一个行动者是谁
        board = self.env.board               # shape = (6,7)
        active_mark = self.env.active_mark   # 1 或 2

        # 拼第 0 通道：把 board==1 的位置设为 1，其余都 0
        board_player1 = (board == 1).astype(np.float32)  # shape (6,7), float32

        # 拼第 1 通道：把 board==2 的位置设为 1，其余都 0
        board_player2 = (board == 2).astype(np.float32)  # shape (6,7), float32

        # 拼第 2 通道：整个板子填充当前行动者对应的符号：如果 active_mark==1 填 1.0，否则填 -1.0
        mark_val = 1.0 if active_mark == 1 else -1.0
        board_to_play = np.full((self.nrows, self.ncols), mark_val, dtype=np.float32)

        # 最终把它们合并成 (3, 6, 7)
        obs = np.stack([board_player1, board_player2, board_to_play], axis=0)
        return obs  # dtype float32, shape (3,6,7)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        执行一步。MuZero 旧环境里 step 的返回是 (obs, reward, done)。
        同理，我们调用新环境的 step，然后重新组观测：
        """
        next_obs_rel, reward, done, info = self.env.step(action)

        # 得到新内部棋盘与当前行动者
        board = self.env.board
        active_mark = self.env.active_mark

        # 同 reset() 里的逻辑，把绝对 board → 三通道 (3,6,7)
        board_player1 = (board == 1).astype(np.float32)
        board_player2 = (board == 2).astype(np.float32)
        mark_val = 1.0 if active_mark == 1 else -1.0
        board_to_play = np.full((self.nrows, self.ncols), mark_val, dtype=np.float32)
        obs = np.stack([board_player1, board_player2, board_to_play], axis=0)

        # 返回 (obs, reward, done)。MuZero 旧环境把 reward 乘 10 放大，这里我们也沿用：
        return obs, reward * 10.0, done

    def to_play(self) -> int:
        """
        MuZero 旧版的 players 是 [0,1]，0 代表第一个行动者 (player1)，1 代表第二个 (player2)。
        现在 internal active_mark 是 1 or 2：
          - 如果是 1，就返回 0
          - 如果是 2，就返回 1
        """
        return 0 if self.env.active_mark == 1 else 1

    def legal_actions(self) -> list:
        """
        返回一轮里所有合法的下棋列，取法同旧版 Connect4.get_legal_actions()。
        新环境本身并没有暴露 legal_actions 接口，但我们知道：
          - 如果某列最顶部 (board[5][col]) 是 0，就表示那列还能下子
        """
        board = self.env.board  # shape (6,7)
        legal = []
        for col in range(self.ncols):
            # 注意：board[i][j] = 0 表示空格；i 从 0 到 5，顶层是索引 5
            if board[self.nrows - 1][col] == 0:
                legal.append(col)
        return legal

    def render(self):
        """
        渲染功能。把内部 board 打印出来，x 表示 player2，o 表示 player1。
        同旧环境的格式：
            - 先从最上面一行开始往下
            - 用 “o” 表示 player1, “x” 表示 player2
            - 最后一行打印 “1 2 3 4 5 6 7”
            - 然后 pause，等用户按回车
        """
        board = self.env.board
        lines = []
        for row in board[::-1]:
            line = []
            for col in row:
                if col == 1:
                    ch = "o"   # player1
                elif col == 2:
                    ch = "x"   # player2
                else:
                    ch = " "
                line.append(ch)
            lines.append(" ".join(line))
        lines.append("-" * (2 * self.ncols))  # 例如 "------------"
        lines.append(" ".join(str(i + 1) for i in range(self.ncols)))
        print("\n".join(lines))
        input("Press enter to take a step ")

    def human_to_action(self) -> int:
        """
        人机交互模式时，让用户在 Terminal 输入一个合法动作 (列号)，并检查其合法性。
        返回一个 int（0~6）。
        """
        to_play = self.to_play()
        choice = input(f"Enter the column to play for the player {to_play} (0-{self.ncols - 1}): ")
        # 验证用户输入
        while (not choice.isnumeric()) or (int(choice) not in self.legal_actions()):
            choice = input("Enter another column (e.g. 0,1,…): ")
        return int(choice)

    def expert_agent(self) -> int:
        """
        MuZero 可以和一个“expert”对手比赛。旧环境里 expert_agent() 是
        通过 Connect4.expert_action() 硬编码一个策略。
        新环境里没有现成的 expert——这里先简单用随机合法动作代替：
        """
        legal = self.legal_actions()
        return np.random.choice(legal) if len(legal) > 0 else 0

    def action_to_string(self, action_number: int) -> str:
        """
        把动作编号转换成可读字符串，供 MuZero 在 human_vs_ai 模式时打印。
        """
        return f"Play column {action_number + 1}"
