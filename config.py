import torch

class Config:
    """Data and placement config: """
    train_dir = '/host-dir/deep_game'
    ckpt_count = 3
    eager = False

    label = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """Training and task selection config: """
    train_steps = 1000000
    warmup = 0.0
    learning_rate = 0.0005
    task = 'hex'
    swap_allowed = True