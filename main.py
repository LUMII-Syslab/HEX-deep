import time

from config import Config
from model.hex3 import HexGame  # Use this when working on the Hex model
#from model.tic_tac import HexGame  # Use this when working on the Tic-Tac-Toe model
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.evaluate_against_sota import evaluate_against_sota
from utils.measure import Timer
import numpy as np

# When loading the previous model, make sure that the previous model was trained on the
#   same task (that is, Hex or Tic-Tac-Toe)
load_prev = False
model_path = "./saved_model"

def main():
    writer = SummaryWriter()
    model = HexGame(summary_writer=writer)
    start_train_step = 0
    if load_prev: start_train_step = load(model)
    train(model, writer, start_train_step)
def save(model, train_step):
    save_dict = {
        "train_step": train_step,
        "model": model.state_dict(),
        "optimizer1": model.optimizer1.state_dict(),
    }

    if model.name == "Tic-Tac-Toe":
        save_dict["optimizer2"] = model.optimizer2.state_dict()

    torch.save(save_dict, model_path)

def load(model):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.optimizer1.load_state_dict(checkpoint['optimizer1'])
    if model.name == "Tic-Tac-Toe":
        model.optimizer2.load_state_dict(checkpoint['optimizer2'])
    train_step = checkpoint['train_step']
    print("model restored at timestep", train_step)
    # Plus one, so it doesn't load from 14 and then train on 14 again
    return train_step + 1

def train(model, writer,start_train_step=0):
    model.copy_weights()
    model.save_to_pool()
    timer = Timer(start_now=True)
    pool_step_limit = 10
    pool_step=0

    for train_step in range(start_train_step,start_train_step+Config.train_steps):
        loss = model.train_step(train_step)
        pool_step+=1

        if int(train_step) % 100 == 0:
            writer.add_scalar("loss", loss, train_step)
            print(f"{int(train_step)}. step;\tloss: {loss:.5f};\ttime: {timer.lap():.3f}s")

            # with tf.name_scope("variables"):
            #     with writer.as_default():
            #         for var in model.trainable_variables:  # type: tf.Variable
            #             tf.summary.histogram(var.name, var, step=int(ckpt.step))
            test_loss1, test_loss2 = model.predict_step(train_step)
            writer.add_scalar("testloss/green", test_loss1, train_step)
            writer.add_scalar("testloss/red", test_loss2, train_step)
            writer.flush()

        if int(train_step) % 200 == 0:
            model.save_to_pool()

        if int(train_step) % 50 == 0:
            alpha = 1. / (train_step / 50 + 1)
            model.average_weights(alpha)

        # if pool_step % pool_step_limit == pool_step_limit-1:
        #     alpha = np.sqrt(1./(train_step/50+1))
        #     model.average_weights(alpha)
        #     model.save_to_pool()
        #     pool_step_limit += 10
        #     print("move_old", pool_step_limit)
        #     pool_step=0

        # """ Evaluate current model against the SOTA solver """
        if train_step % 1000 == 999:
            evaluate_against_sota(model, train_step, writer)

        if int(train_step) % 1000 == 999:
            save(model, train_step)
            print(f"Saved checkpoint for step {train_step}: {model_path}")


if __name__ == '__main__':
    current_date = time.strftime("%y_%m_%d_%T", time.gmtime(time.time()))
    label = "_" + Config.label if Config.label else ""
    Config.train_dir = Config.train_dir + "/" + Config.task + "_" + current_date + label

    main()
