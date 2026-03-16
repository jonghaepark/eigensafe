import argparse


def parser_args():
    parser = argparse.ArgumentParser(description='PyTorch EigenSafe Args')

    # Exp
    parser.add_argument('--exp_name', default="exp0301-eigen-lambda1200-lander-1",
                        help='name of experiment')
    
    # Env
    parser.add_argument('--env_name', default="LunarLander-safety",
                        help='Mujoco Gym environment (default: Halfcheetah-run-low-v5, Hopper-run-high-v5, Ant-ball-v5, LunarLander-safety)')
    parser.add_argument('--seed', type=int, default=970, metavar='N',
                        help='random seed (default: 123)')

    # DDPG
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: True)')
    parser.add_argument('--alpha', type=float, default=0.01, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')


    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='G',
                        help='learning rate (default: 0.0001)')

    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')

    # Buffer RL
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='minibatch size of batch (default: 256)')
    parser.add_argument('--replay_size', type=int, default=50000, metavar='N',
                        help='total capacity of replay buffer (default: 1000000)')
    
    # Network
    parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                        help='hidden size (default: 1024)')

    # Largransian
    parser.add_argument('--lambda_value', type=float, default=1200, metavar='G',
                        help='lambda_value for psi (default: 10)')
    parser.add_argument('--epsilon', type=float, default=1e-3, metavar='G',
                        help='epsilon for psi (default: 1e-3)')

    # Target gamma
    parser.add_argument('--gamma_target', type=float, default=1.0, metavar='G',
                        help='target gamma for psi (default: 1)')
    
    # Update to data(UTD) ratio
    parser.add_argument('--episodes_per_epoch', type=int, default=10, metavar='N',
                        help='episodes_per_epoch')
    parser.add_argument('--gradient_steps_per_epoch', type=int, default=64, metavar='N',
                        help='model updates per epoch (default: 1)')

    # Max/start step
    parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')

    # Logging period
    parser.add_argument('--eval_epoch_ratio', type=int, default=100, metavar='N',
                        help='model updates per epoch (default: 1)')
    parser.add_argument('--save_epoch_ratio', type=int, default=500, metavar='N',
                        help='model updates per epoch (default: 1)')
    
    # Max episode
    parser.add_argument('--num_episodes', type=int, default=80000, metavar='N',
                        help='maximum number of episodes (default: 1000)')
    
    # Device
    parser.add_argument('--cuda', action="store_false",
                        help='run on CUDA (default: True)')
    
    return parser.parse_args()