# File: src/agents/baselines/MuZeroAgent.py

import os
import sys
import numpy as np
import torch

# 确保项目根目录在 PYTHONPATH 中，以便能 import `src.muzero` 模块
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from src.muzero.models import MuZeroNetwork
from src.muzero.self_play import MCTS, SelfPlay
from src.muzero.games.connect4_new import MuZeroConfig   # 你的 MuZeroConfig
from src.wrapped_connect4 import WrappedConnect4         # 你的状态转换器（默认构造即可）
from src.agents.agent import Agent


class MuZeroAgent(Agent):
    """
    MuZeroAgent: 在 ConnectGameEnv 环境下使用 MuZeroNetwork + MCTS 做推理。
    - __init__：直接创建 MuZeroNetwork 并加载权重，无需完整 MuZero 分布式框架。
    - get_action：把 (6×7) 观测转换为 (3×6×7)，用 MCTS 进行搜索，并返回访问次数最高的动作。
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        super().__init__(name="MuZeroAgent")

        # 尽量用 GPU，否则回退到 CPU
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ————————————————
        # 1) 载入 checkpoint：必须包含 "weights"，可选包含 "config"
        #    注意 torch.load 默认 weights_only=True，需要显式传 weights_only=False
        # ————————————————
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 如果 checkpoint 里有 config，就直接用；否则用默认 MuZeroConfig()
        self.config = checkpoint.get("config", None)
        if self.config is None:
            self.config = MuZeroConfig()

        # ————————————————
        # 2) 直接创建一个 MuZeroNetwork 实例，并用 strict=False 加载权重
        # ————————————————
        self.network = MuZeroNetwork(self.config)

        # load_state_dict(..., strict=False) 可以跳过意外的键或缺失的键
        load_result = self.network.load_state_dict(checkpoint["weights"], strict=False)
        # 如果想追踪缺失/意外的键，可以打开下面的打印：
        if hasattr(load_result, "missing_keys") and load_result.missing_keys:
            print(f"[MuZeroAgent] Warning: missing keys when loading weights: {load_result.missing_keys}")
        if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
            print(f"[MuZeroAgent] Warning: unexpected keys in checkpoint: {load_result.unexpected_keys}")

        self.network.to(self.device)
        self.network.eval()

        # ————————————————
        # 3) 如果 WrappedConnect4 有默认构造，就直接用；如果它没有默认构造，需要你把合适的参数传进去
        #    （但本示例中，我们在 get_action() 里已经手动把 (6×7) → (3×6×7)，并未真正用到 state_wrapper）
        # ————————————————
        try:
            self.state_wrapper = WrappedConnect4()
        except TypeError:
            # 如果 WrappedConnect4 需要别的参数，请在这里自行添加
            self.state_wrapper = None

        # 推理阶段用的 MCTS 模拟次数
        self.num_simulations = self.config.num_simulations

        # MuZeroAgent 不允许走非法动作（我们在 get_action 中会对非法动作做屏蔽）
        self.allow_illegal_actions = False


    def get_action(self,
                   observation: np.ndarray,
                   to_play: int,
                   legal_actions: list) -> int:
        """
        :param observation:   np.ndarray, shape=(6,7), float32，值 ∈ { 1, 0, -1 }，
                              “1” 表示 active player 的棋子，“-1” 表示对手的棋子。
        :param to_play:       int，0 或 1，表示当前轮到谁走棋
                              （在 ConnectGameEnv 中，active_mark == 1 对应 to_play=0，active_mark==2 对应 to_play=1）。
        :param legal_actions: List[int]，合法动作（列号列表），表示哪些列还能下子。
        :return: int，选出的列号（0..6），必须在 legal_actions 中。
        """

        # — Step 1:
        # 把原始 (6×7) 观测手动转换为 MuZero 需要的 (3×6×7)：
        #   ch0 = active player 的 bitboard (值 == 1 的位置设为 1，其余为 0)
        #   ch1 = opponent      的 bitboard (值 == -1 的位置设为 1，其余为 0)
        #   ch2 = 全盘常数 +1 或 -1，用来标记“当前是谁在下”，to_play=0→+1，to_play=1→-1
        obs_np = observation.astype(np.float32)   # shape = (6,7)
        ch0 = (obs_np == 1.0).astype(np.float32)   # active player 的位置
        ch1 = (obs_np == -1.0).astype(np.float32)  # opponent 的位置
        to_play_marker = 1.0 if to_play == 0 else -1.0
        ch2 = np.full_like(obs_np, fill_value=to_play_marker, dtype=np.float32)

        # 堆叠成 shape = (3,6,7)
        stacked = np.stack([ch0, ch1, ch2], axis=0)

        # — Step 2:
        # 用 MCTS 搜索。MCTS.run 内部会自动把 NumPy 转为 tensor，
        # 并调用 network.initial_inference() / recurrent_inference() 生成搜索树。
        root, _ = MCTS(self.config).run(
            model=self.network,            # 传入 MuZeroNetwork 实例
            observation=stacked,           # (3,6,7) 的 NumPy array
            legal_actions=legal_actions,   # 当前合法动作（列号列表）
            to_play=to_play,               # 0 或 1
            add_exploration_noise=False,   # 推理阶段不要 Dirichlet 噪声
            num_simulations=self.num_simulations
        )

        # — Step 3:
        # 从根节点的孩子里选访问次数最多的动作 (temperature=0)
        best_action = SelfPlay.select_action(root, temperature=0.0)
        return int(best_action)
