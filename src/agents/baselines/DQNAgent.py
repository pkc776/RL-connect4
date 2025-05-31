# 檔案路徑：src/agents/dqn_agent.py

import json
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.agents.agent import Agent  # 請確保這個基底類別存在且可 import


class DynamicDQNNet(nn.Module):
    """
    動態從 cnet128.json 讀取 conv_block、fc_block、first_head、second_head，
    建立整支網路。forward 時把 (batch,6,7) → (batch,2,6,7)，再依序 conv→fc→head。
    """

    def __init__(self, arch_json_path: str):
        """
        :param arch_json_path: 指向 cnet128.json 的檔案路徑
        """
        super().__init__()

        # 1. 讀 JSON
        if not os.path.isfile(arch_json_path):
            raise FileNotFoundError(f"[DynamicDQNNet] 找不到架構定義檔：{arch_json_path}")

        with open(arch_json_path, "r", encoding="utf-8") as f:
            arch = json.load(f)

        # ------ 解析 conv_block ------
        # JSON 裡 "conv_block": [[128,4,0], "relu", [128,2,0], "relu"]
        # 每個 list 代表一層 Conv，格式為 [out_channels, kernel_size, padding]
        # 字串 "relu" 代表插入一個 nn.ReLU()
        conv_layers = []
        in_channels = 2  # 輸入固定為 2 通道 (active_mask, opp_mask)
        for idx, layer_cfg in enumerate(arch.get("conv_block", [])):
            if isinstance(layer_cfg, list):
                # [out_channels, kernel_size, padding]
                out_ch = layer_cfg[0]
                ksize  = layer_cfg[1]
                pad    = layer_cfg[2]
                conv_layers.append(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=out_ch,
                              kernel_size=ksize,
                              stride=1,
                              padding=pad,
                              bias=True)
                )
                # 更新下一層的 in_channels
                in_channels = out_ch
            elif isinstance(layer_cfg, str) and layer_cfg.lower() == "relu":
                conv_layers.append(nn.ReLU())
            else:
                raise ValueError(f"[DynamicDQNNet] conv_block 裡遇到未知元素：{layer_cfg}")

        self.conv_block = nn.Sequential(*conv_layers)
        # 到此，假設輸入是 (batch,2,6,7)，第一次 conv (kernel=4,pad=0)→(batch,128,3,4)，
        # 第二次 conv (kernel=2)→(batch,128,2,3)。

        # ------ 解析 fc_block ------
        # JSON 裡 "fc_block": [128, "relu"]
        # 意味著：Linear(128*2*3, 128) + ReLU
        fc_layers = []
        fc_cfg = arch.get("fc_block", [])
        if len(fc_cfg) >= 1:
            # 第一個元素應該是 out_features (128)
            out_f = fc_cfg[0]
            # conv_block 最後的輸出 shape = (batch, 128, 2, 3)，flatten 後 = 128*2*3
            in_f = 128 * 2 * 3
            fc_layers.append(nn.Linear(in_f, out_f, bias=True))

            # 如果後面有 "relu"，就加 ReLU
            if len(fc_cfg) > 1 and isinstance(fc_cfg[1], str) and fc_cfg[1].lower() == "relu":
                fc_layers.append(nn.ReLU())
        else:
            raise ValueError("[DynamicDQNNet] fc_block 配置不可為空。")

        self.fc_block = nn.Sequential(*fc_layers)

        # ------ 解析 first_head ------
        # JSON 裡 "first_head": [128, "relu", 7]
        # 意味著：Linear(128, 128) + ReLU + Linear(128, 7)
        first_head_layers = []
        fh_cfg = arch.get("first_head", [])
        if len(fh_cfg) == 3:
            hidden_dim = fh_cfg[0]
            if isinstance(fh_cfg[1], str) and fh_cfg[1].lower() == "relu":
                out_dim = fh_cfg[2]  # 7
                # 隱藏層
                first_head_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
                first_head_layers.append(nn.ReLU())
                # 輸出層
                first_head_layers.append(nn.Linear(hidden_dim, out_dim, bias=True))
            else:
                raise ValueError("[DynamicDQNNet] first_head 第二個元素應為 'relu'。")
        else:
            raise ValueError("[DynamicDQNNet] first_head 必須有三個元素：[hidden, 'relu', output]。")

        self.first_head = nn.Sequential(*first_head_layers)

        # ------ 解析 second_head ------
        # JSON 裡 "second_head": [128, "relu", 1]
        # 這邊我們也照同樣邏輯建出來，但 DQNAgent 通常只用 first_head 來算 Q 值。
        second_head_layers = []
        sh_cfg = arch.get("second_head", [])
        if len(sh_cfg) == 3:
            hidden_dim = sh_cfg[0]
            if isinstance(sh_cfg[1], str) and sh_cfg[1].lower() == "relu":
                out_dim = sh_cfg[2]  # 1
                # 隱藏層
                second_head_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
                second_head_layers.append(nn.ReLU())
                # 輸出層
                second_head_layers.append(nn.Linear(hidden_dim, out_dim, bias=True))
            else:
                raise ValueError("[DynamicDQNNet] second_head 第二個元素應為 'relu'。")
        else:
            # 若 JSON 沒提供 second_head，我們這邊可以跳過不建（但本例有提供）
            second_head_layers = []

        self.second_head = nn.Sequential(*second_head_layers) if second_head_layers else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (batch_size, 6, 7)，值為 {1, 0, -1}。
        :return: Tensor of shape (batch_size, 7)，第一個 head (Q-values) 的輸出。
        """
        # 1. 把 (batch,6,7) → 兩通道 (batch,2,6,7)
        active_mask = (x == 1).float().unsqueeze(1)    # (batch,1,6,7)
        opp_mask    = (x == -1).float().unsqueeze(1)   # (batch,1,6,7)
        x_two = torch.cat([active_mask, opp_mask], dim=1)  # (batch,2,6,7)

        # 2. conv_block
        y = self.conv_block(x_two)   # e.g. (batch,128,2,3)

        # 3. flatten + fc_block
        y = y.view(y.size(0), -1)    # (batch, 128*2*3 = 768)
        y = self.fc_block(y)         # (batch,128)

        # 4. first_head → Q-values (batch,7)
        q = self.first_head(y)       # (batch,7)
        return q

        # 如果你日後要用 second_head (value 頭)，可以另外呼叫：
        # v = self.second_head(y)    # (batch,1)


class DQNAgent(Agent):
    """
    DQN Agent：
    1) 建立 DynamicDQNNet，從 JSON 自動生成 conv_block/ fc_block/ first_head/ second_head。
    2) 用 supervised_cnet128.pt 把 conv_block 的預訓練權重 load 進來，並凍結 conv_block。
    3) 再用 checkpoint_path (DQN 訓練後的整組權重) load_state_dict(strict=False) 覆蓋 fc_block + first_head (+ second_head)。
    4) get_action 時只用 first_head 的 Q-values 遮罩非法動作後 argmax。
    """

    def __init__(
        self,
        arch_json_path: str,
        pretrained_conv_path: str,
        checkpoint_path: str,
        device: str = "cpu"
    ):
        """
        :param arch_json_path:       cnet128.json 的路徑
        :param pretrained_conv_path: supervised_cnet128.pt 的路徑（僅含 conv_block 權重）
        :param checkpoint_path:       DQN 訓練好的完整權重檔（state_dict 或 {"model_state_dict":...}）
        :param device:               "cpu" 或 "cuda"
        """
        super().__init__(name="DQNAgent")

        # 1. 設定 device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 2. 動態建立 network 架構
        self.net = DynamicDQNNet(arch_json_path).to(self.device)

        # 3. 載入預訓練的 conv_block 權重，然後凍結 conv_block
        if not os.path.isfile(pretrained_conv_path):
            raise FileNotFoundError(f"[DQNAgent] 找不到預訓練卷積權重：{pretrained_conv_path}")

        ckpt_conv = torch.load(pretrained_conv_path, map_location=self.device)
        # 如果裡面包了 {"model_state_dict": ...}，就取出來
        if isinstance(ckpt_conv, dict) and "model_state_dict" in ckpt_conv:
            ckpt_conv = ckpt_conv["model_state_dict"]

        # 取得目前 net 的 state_dict
        own_state = self.net.state_dict()
        # 只把以 "conv_block." 開頭的那些 key copy over
        for name, param in ckpt_conv.items():
            if name.startswith("conv_block."):
                if name in own_state and param.size() == own_state[name].size():
                    own_state[name].copy_(param)
                else:
                    print(f"[DQNAgent WARNING] conv 層權重 key「{name}」或尺寸 {param.size()} 不吻合，跳過。")

        # 4. 凍結 conv_block 的所有參數
        for n, p in self.net.named_parameters():
            if n.startswith("conv_block."):
                p.requires_grad = False

        # 5. 載入 DQN 訓練好的 checkpoint，strict=False 讓非 conv_block 的部分被覆寫
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"[DQNAgent] 找不到 DQN checkpoint：{checkpoint_path}")

        ckpt_dqn = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt_dqn, dict) and "model_state_dict" in ckpt_dqn:
            ckpt_dqn = ckpt_dqn["model_state_dict"]

        # 由於 conv_block 已經是凍結的預訓練權重，這裡 strict=False 代表若 checkpoint 裡也有 conv_block 權重
        # 也不會覆寫（因為之前已經把 requires_grad=False），但其他 fc_block / first_head / second_head 都會被灌進來
        self.net.load_state_dict(ckpt_dqn, strict=False)

        # 6. 統一進入 eval() 模式（inference）
        self.net.eval()

        # 7. DQNAgent 不允許非法動作
        self.allow_illegal_actions = False

    def get_action(self, observation: np.ndarray, to_play: int, legal_actions: List[int]) -> int:
        """
        :param observation:   shape (6,7)，值為 {1: active player, 0: empty, -1: opponent}。
        :param to_play:       哪個玩家在下（1 或 2），只是做為紀錄，實際 network 只用 observation 裡的 +1/−1。
        :param legal_actions: List[int]，範圍 0..6，代表哪些 column 還能下。
        :return:              int，在 legal_actions 中使 Q 值最大的那格 column。
        """
        # 1) 把 observation 轉成 tensor，維度 (1,6,7)，dtype float32
        obs_tensor = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0).to(self.device)  # (1,6,7)

        # 2) network forward 得到 Q-values (1,7)
        with torch.no_grad():
            q_tensor = self.net(obs_tensor)          # (1,7)
            q_values = q_tensor.cpu().numpy()[0]     # (7,)

        # 3) 遮罩不合法動作
        illegal = set(range(7)) - set(legal_actions)
        for a in illegal:
            q_values[a] = -float("inf")

        # 4) 回傳 Q 值最大的合法 column
        best_action = int(np.argmax(q_values))
        return best_action
