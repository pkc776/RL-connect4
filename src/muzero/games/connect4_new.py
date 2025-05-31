# 文件路径示例：你的项目根目录/games/connect4_new.py
import os 
import datetime
import numpy
import torch
import sys
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)
from .abstract_game import AbstractGame
from src.wrapped_connect4 import WrappedConnect4  # 确保 Python 能找到 wrapped_connect4.py 的位置
class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        self.max_num_gpus = None

        # Game
        # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (3, 6, 7)
        # Fixed list of all possible actions. You should only edit the length
        self.action_space = list(range(7))
        # List of players. You should only edit the length
        self.players = list(range(2))
        # Number of previous observations and previous actions to add to the current observation
        self.stacked_observations = 0

        # Evaluate
        # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.muzero_player = 0
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        # Self-Play
        # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_workers = 5
        self.selfplay_on_gpu = False
        self.max_moves = 42  # Maximum number of moves if game is not finished before
        self.num_simulations = 10  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.2
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 10

        # Residual Network
        # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.downsample = False
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = [64]
        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = [64]
        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = [64]

        # Fully Connected Network
        self.encoding_size = 32
        # Define the hidden layers in the representation network
        self.fc_representation_layers = []
        # Define the hidden layers in the dynamics network
        self.fc_dynamics_layers = [64]
        # Define the hidden layers in the reward network
        self.fc_reward_layers = [64]
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        # Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[
                                         :-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        # Total number of training steps (ie weights update according to a batch)
        self.training_steps = 1000000
        self.batch_size = 256  # Number of parts of games to train on at each training step
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = 300
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-5  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        # Replay Buffer
        # Number of self-play games to keep in the replay buffer
        self.replay_buffer_size = 10000
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 42
        # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER = False
        # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = 0.5

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = False
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    @property
    def random_move_till_n_action_in_self_play(self):
        return numpy.random.choice([2, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8])

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < self.training_steps * 0.5:
            return 1
        else:
            return 0.5

class Game(AbstractGame):
    """
    MuZero 需要的 Game 接口。我们直接用 WrappedConnect4 进行封装，
    以便 MuZero 其余代码（reset, step, to_play, legal_actions, render 等）都能照常调用。
    """

    def __init__(self, seed=None):
        # 忽略 seed，因为 WrappedConnect4 目前没有用到随机种子。
        # 如果你想让新环境也根据 seed 固定随机初始盘面，可以把 seed 传给 WrappedConnect4。
        self.env = WrappedConnect4()

    def step(self, action: int):
        """
        MuZero 在整棵搜索树里，会用 env.step(action) 获取 (obs, reward, done)。
        这里直接调用 WrappedConnect4.step，返回三元组。
        """
        obs, reward, done = self.env.step(action)
        return obs, reward, done

    def to_play(self) -> int:
        """
        MuZero 在多玩家场景会调用这个接口来判断当前是谁下棋。
        """
        return self.env.to_play()

    def legal_actions(self) -> list:
        """
        MuZero 会用这个接口获取当前局面下可以选的所有动作。
        """
        return self.env.legal_actions()

    def reset(self):
        """
        MuZero 在每局开始时会调用 reset() 获取 initial observation。
        """
        return self.env.reset()

    def render(self):
        """
        MuZero 在 human vs MuZero 或 test 时，如果指定 render=True，
        就会调用这个接口把游戏画面渲染出来。
        """
        return self.env.render()

    def human_to_action(self) -> int:
        """
        MuZero 要做人机对战时，会调用这个接口获取人类玩家的动作。
        """
        return self.env.human_to_action()

    def expert_agent(self) -> int:
        """
        MuZero 在自我对战评估里，如果 config.opponent="expert"，
        就会调用这个接口获取 expert 动作。我们这里随便给个随机，
        你也可以按需要实现一个更强的“expert”逻辑。
        """
        return self.env.expert_agent()

    def action_to_string(self, action_number: int) -> str:
        """
        MuZero 在 human vs MuZero 时，会打印出动作的可读形式。
        """
        return self.env.action_to_string(action_number)
