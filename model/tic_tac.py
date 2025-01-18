import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch.optim.lr_scheduler import LambdaLR

from model.conv_simple import ConvSimple
from torch.utils.tensorboard import SummaryWriter

from model.hex3 import sample_gumbel, softmax_global, softmax_global_soft
from utils.reinmax import reinmax
from config import Config
def randomized_rounding(logits, valid_positions, inner_noise_scale, outer_noise_scale=1., outer_noise_weight=0.0, training=True):
    s = logits.shape
    logits = logits + sample_gumbel(logits.shape) * inner_noise_scale
    x_flat = logits.view([s[0],-1])
    #a, b = reinmax(x_flat,1.0)
    #return a.view(s)

    y_soft = torch.softmax(x_flat, dim=-1)
    if training:
        outer_noise = torch.softmax(sample_gumbel(x_flat.shape) * outer_noise_scale, dim=-1)
        outer_noise = outer_noise * valid_positions.view([s[0],-1])
        outer_noise = outer_noise / (torch.sum(outer_noise, dim=-1, keepdim=True)+1e-8)
        y_soft = y_soft*(1-outer_noise_weight) + outer_noise*outer_noise_weight
    index = y_soft.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(x_flat, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    #ret = y_soft

    return ret.view(s)

class HexGame(nn.Module):

    def __init__(self, feature_maps=256, rounds=4,board_size=3,batch_size=128, summary_writer=None,**kwargs):
    #def __init__(self, feature_maps=256, rounds=20, board_size=7, batch_size=128, summary_writer=None, **kwargs):
        super(HexGame, self).__init__(**kwargs)
        self.name = "Tic-Tac-Toe"
        self.rounds = rounds
        self.board_size = board_size
        self.feature_maps = feature_maps
        self.batch_size = batch_size
        self.move_layers = 3
        self.regul_weight = 0.01#0.2 # initially 0.001
        self.player_model = ConvSimple(4, feature_maps, 1, board_size).to(Config.device)
        self.fixed_player_model = ConvSimple(4, feature_maps, 1, board_size).to(Config.device)
        self.player_model_old = ConvSimple(4, feature_maps, 1, board_size).to(Config.device)
        self.model_pool = []
        def warmup(current_step: int):
            warmup_steps = 5000
            if current_step < warmup_steps:  # current_step / warmup_steps * base_lr
                return float(current_step / warmup_steps)
            else:
                return 1.

        self.optimizer1 = torch.optim.AdamW(self.player_model.parameters(), lr=Config.learning_rate, betas=(0.5, 0.999))
        self.scheduler1 = LambdaLR(self.optimizer1, lr_lambda=warmup)
        self.optimizer2 = torch.optim.AdamW(self.player_model.parameters(), lr=Config.learning_rate, betas=(0.5, 0.999))
        self.scheduler2 = LambdaLR(self.optimizer2, lr_lambda=warmup)

        self.summary_writer = summary_writer

    def calc_regul(self,logits,occupied):
        #logits = logits - torch.mean(logits, dim=[1,2,3], keepdim=True)
        regul_loss = torch.sum(torch.square(logits)*(1-occupied), dim=[1,2,3])
        return regul_loss.mean()

    def forward(self, train_player_id, player1_random_prob = 0.0, rounding=False, training=True, global_step=0):
        player2_random_prob = 0.0
        total_loss1 = 0.
        total_loss2 = 0.
        player1_probs = torch.zeros([self.batch_size,1,self.board_size, self.board_size], device=Config.device)
        player2_probs = torch.zeros([self.batch_size,1,self.board_size, self.board_size], device=Config.device)
        move_order = torch.zeros([self.batch_size, 1, self.board_size, self.board_size], device=Config.device)
        move1 = torch.zeros([self.batch_size, 1, self.board_size, self.board_size], device=Config.device)
        move2 = torch.zeros([self.batch_size, 1, self.board_size, self.board_size], device=Config.device)
        val_str = "val" if not training else str(train_player_id) + "id"
        regul_loss = 0.

        if train_player_id == 0:
            model1 = self.player_model
            model2 = self.fixed_player_model
            if random.random() < 0.9 and training:
                #model2 = self.player_model_old
                model2 = random.choice(self.model_pool)
        else:
            model1 = self.fixed_player_model
            model2 = self.player_model
            if random.random() < 0.9 and training:
                model1 = random.choice(self.model_pool)

        player1_wins = torch.zeros([self.batch_size], device=Config.device)
        player2_wins = torch.zeros([self.batch_size], device=Config.device)
        win_time = torch.zeros([self.batch_size], device=Config.device)

        for step in range(self.rounds):
            noise_scale = 1.0# + 3.0 * torch.square(torch.rand((self.batch_size, 1, 1, 1),device=Config.device))
            mul_noise_scale = 0.0
            op_noise_scale = 0.0#6.0 * torch.square(torch.rand((self.batch_size, 1, 1, 1),device=Config.device))

            move1_logits = model1(torch.cat([player1_probs, player2_probs,1-player1_probs-player2_probs, move2], dim=1))
            regul_loss1 = self.calc_regul(move1_logits, player1_probs + player2_probs)
            # if train_player_id==1:
            #     move1_logits = move1_logits.detach()
            #logits_norm1 = torch.mean(torch.abs(move1_logits), dim=[1,2,3], keepdim=True)
            logits_norm1 = torch.var(move1_logits, dim=[1, 2, 3],correction=0, keepdim=True)
            move1_logits_org = move1_logits
            noise_scale1 = mul_noise_scale*logits_norm1+noise_scale+op_noise_scale*train_player_id #use more noise for the opponent

            if rounding:
                move1 = self.discretize(move1_logits)
            else:
                move1 = randomized_rounding(move1_logits,1-player1_probs-player2_probs, noise_scale1, training=training)
            game_over = player1_wins+player2_wins
            win_time = win_time + (1 - game_over)
            move1 = move1*(1-game_over).view([self.batch_size, 1, 1, 1]) # don't move if game is over
            player1_probs_prev = player1_probs

            if random.random() < player1_random_prob:
               move1, player1_probs = self.random_move(player1_probs, player2_probs, player1_wins+player2_wins)
            else:
                move1,player1_probs = self.move(move1, player1_probs, player2_probs)
            move_order = move_order + move1*(2*step)
            our_win_reward = self.win_reward(player1_probs, training)
            #our_win_reward = our_win_reward*(1-player2_wins)
            player1_wins = torch.maximum(player1_wins, our_win_reward)
            #our_win_reward = torch.mean(our_win_reward)
            path_map = softmax_global_soft(move1_logits_org)[:,0,:,:].detach().cpu().numpy()

            #move2_logits,state2 = model2(torch.cat([player2_probs, player1_probs,1-player1_probs-player2_probs, move1], dim=1),state2)
            move2_logits = model2(torch.cat([player2_probs, player1_probs, 1 - player1_probs - player2_probs, move1], dim=1))
            regul_loss2 = self.calc_regul(move2_logits,player1_probs+player2_probs)
            # if train_player_id==0:
            #     move2_logits = move2_logits.detach()

            #logits_norm2 = torch.mean(torch.abs(move2_logits), dim=[1,2,3], keepdim=True)
            logits_norm2 = torch.var(move2_logits, dim=[1, 2, 3], correction=0, keepdim=True)
            move2_logits_org = move2_logits
            noise_scale2 = mul_noise_scale*logits_norm2 + noise_scale + op_noise_scale * (1-train_player_id)  # use more noise for the opponent

            if rounding:
                move2 = self.discretize(move2_logits)
            else:
                move2 = randomized_rounding(move2_logits,1-player1_probs-player2_probs, noise_scale2, training=training)
            game_over = player1_wins + player2_wins
            win_time = win_time + (1 - game_over)
            move2 = move2 * (1 - game_over).view([self.batch_size, 1, 1, 1])  # don't move if game is over
            player2_probs_prev = player2_probs

            if random.random() < player2_random_prob:
                move2, player2_probs = self.random_move(player2_probs, player1_probs, player1_wins+player2_wins)
            else:
                move2,player2_probs = self.move(move2, player2_probs, player1_probs)
            move_order = move_order + move2 * (2 * step+1)

            op_win_reward = self.win_reward(player2_probs, training)
            #op_win_reward = op_win_reward*(1-player1_wins)
            player2_wins = torch.maximum(player2_wins, op_win_reward)
            #op_win_reward = torch.mean(op_win_reward)
            op_path_map=softmax_global_soft(move2_logits_org)[:,0,:,:].detach().cpu().numpy()

            #regul_loss1 = torch.var(move1_logits_org, correction=0)

            total_loss1 = total_loss1 + regul_loss1*self.regul_weight/(step+1)
            total_loss2 = total_loss2 + regul_loss2*self.regul_weight/(step+1)

            regul_loss = regul_loss+regul_loss1+regul_loss2

            if not training or global_step % 20 == 1:
                move2_soft = softmax_global_soft(move2_logits_org)
                move1_soft = softmax_global_soft(move1_logits_org)
                #_, player1_probs_soft = self.move(move1_soft, player1_probs_prev, player2_probs_prev)
                #_,player2_probs_soft = self.move(move2_soft, player2_probs_prev, player1_probs)
                self.show_image(player1_probs,player2_probs_prev, val_str+'probs/' + str(step*2), path_map, global_step)
                self.show_image(player1_probs, player2_probs, val_str+'probs/' + str(step * 2+1), op_path_map, global_step)
                self.show_image(move1_soft,torch.zeros_like(move2), val_str+'move3/' + str(step * 2), None, global_step)
                self.show_image(torch.zeros_like(move1), move2_soft, val_str+'move3/' + str(step * 2 + 1), None, global_step)
                self.summary_writer.add_histogram('move/' + str(step * 2 + 1), move1_logits_org[0:1, ...], global_step=global_step)

        # after loop
        opponont_weight = 5.0  # 0.1/self.board_size
        # player1_loss = (1-our_win_reward)*torch.detach(1-our_win_reward)
        # player2_loss = (1 - op_win_reward)*torch.detach(1 - op_win_reward)
        # player1_loss = torch.square(1-our_win_reward).mean()
        # player2_loss = torch.square(1 - op_win_reward).mean()
        # player1_loss = (1 - player1_probs).mean()*0.1
        # player2_loss = (1 - player2_probs).mean()*0.1
        player1_loss = -self.win_loss(player1_probs, training).mean()
        player2_loss = -self.win_loss(player2_probs, training).mean()

        reward_weight = win_time/(self.rounds*2)

        total_loss1 = total_loss1 + player1_loss + opponont_weight * op_win_reward.mean()
        total_loss2 = total_loss2 + player2_loss + opponont_weight * our_win_reward.mean()

        if not training or global_step % 20 == 1:
            #self.summary_writer.add_scalar("loss/all"+val_str,total_loss1+total_loss2, global_step)
            self.summary_writer.add_scalar("loss/green"+val_str, total_loss1, global_step)
            self.summary_writer.add_scalar("loss/red"+val_str, total_loss2, global_step)
            self.summary_writer.add_scalar("regul", regul_loss, global_step)
            self.summary_writer.add_scalar("win/player1"+val_str, player1_wins.mean(), global_step)
            self.summary_writer.add_scalar("win/player2"+val_str, player2_wins.mean(), global_step)
            # self.summary_writer.add_histogram("path/logits", paths2[0:10, ...], global_step)
            # self.summary_writer.add_histogram("path/pred", paths_prediction[0:10, ...], global_step)
            self.summary_writer.add_histogram('win_time', win_time, global_step=global_step)


        return total_loss1, total_loss2

    def show_image(self, probs,other_probs, name, path_mask, globalstep, probfunc = torch.sqrt):
        image = probfunc(probs[0,:,:,:]).detach().cpu().numpy()
        image1 = probfunc(other_probs[0,:,:,:]).detach().cpu().numpy()
        if path_mask is not None:
            path_mask = np.sqrt(path_mask[0:1,:,:])
        else:
            path_mask = np.zeros([1,self.board_size, self.board_size])

        image = np.concatenate([image1, image,path_mask], axis=0)
        # image = np.repeat(image, 2, axis=2)
        #
        # image = np.pad(image, [[0,0],[0,0],[0,self.board_size-1]],'constant', constant_values=0.5)
        # # shear x
        # for i in range(image.shape[1]):
        #     image[:,i, :] = np.roll(image[:,i, :], i)
        # # image = tfa.image.shear_x(tf.cast(image[0,:,:,:]*255, tf.uint8), -1., 128)
        # image = np.repeat(image, 2, axis=1)
        image = (image*255).astype(np.uint8)
        self.summary_writer.add_image(name, image, globalstep)

    def move(self, move_probs, our_probs, opponent_probs, normalize=False):
        remaining_probs = 1-(our_probs+opponent_probs)
        move_probs = move_probs*remaining_probs
        if normalize:
            move_probs = move_probs / (torch.sum(move_probs, dim=[1,2,3], keepdim=True) + 1e-6)
            move_probs = torch.minimum(move_probs, remaining_probs)
        return move_probs, our_probs+move_probs
    def discretize(self, move):
        randomized_probs = move.view([self.batch_size, self.board_size * self.board_size])
        move_index = torch.argmax(randomized_probs, dim=-1)  # choose one of the empty positions
        move_mask = torch.nn.functional.one_hot(move_index, self.board_size * self.board_size)
        move_mask = move_mask.view([self.batch_size, 1,self.board_size, self.board_size])
        return move_mask

    def random_move(self, player1_probs, player2_probs, game_over):
        move_mask_all = torch.zeros([self.batch_size, 1,self.board_size,self.board_size], device = player1_probs.device)

        # choose random move
        for trials in range(5):
            player_probs = player1_probs + player2_probs
            randomized_probs = 1-player_probs + torch.rand(player_probs.shape, device = player_probs.device)
            move_mask = self.discretize(randomized_probs)
            move_sum = torch.sum(move_mask_all, dim=[2, 3], keepdim=True)
            move_mask = move_mask*(1 - move_sum)
            move_mask = move_mask * (1 - game_over).view([self.batch_size, 1, 1, 1])  # don't move if game is over
            move_mask, player1_probs = self.move(move_mask, player1_probs, player2_probs, normalize=False)
            move_mask_all = move_mask_all+move_mask
            move_sum = torch.sum(move_mask_all)
            if move_sum >= self.batch_size * 0.95: break

        return move_mask_all, player1_probs


    def win_reward(self, probs, training):
        probs = probs[:, 0, :, :]
        r1 = torch.sum(torch.prod(probs, dim=2), dim=1)
        r2 = torch.sum(torch.prod(probs, dim=1), dim=1)
        #r3 = probs[:,0,0]*probs[:,1,1]*probs[:,2,2]#torch.eye(self.board_size)
        r3 = torch.prod(torch.diagonal(probs, dim1=1, dim2=2), dim=1)
        #r4 = probs[:, 2, 0] * probs[:, 1, 1] * probs[:, 0, 2]  # torch.eye(self.board_size)
        r4 = torch.prod(torch.diagonal(torch.flip(probs,dims=[-1]), dim1=1, dim2=2), dim=1)
        reward = r1+r2+r3+r4
        return reward

    def win_loss(self, probs, training):
        probs = probs[:, 0, :, :]
        exponent=2.0
        r1 = torch.sum(torch.exp((torch.sum(probs, dim=2)-self.board_size)*exponent), dim=1)
        r2 = torch.sum(torch.exp((torch.sum(probs, dim=1) - self.board_size) * exponent), dim=1)
        #r2 = torch.sum(torch.prod(probs, dim=1), dim=1)
        r3 = torch.exp((torch.sum(torch.diagonal(probs, dim1=1, dim2=2), dim=1)-self.board_size)*exponent)
        #r4 = probs[:, 2, 0] * probs[:, 1, 1] * probs[:, 0, 2]  # torch.eye(self.board_size)
        r4 = torch.exp((torch.sum(torch.diagonal(torch.flip(probs,dims=[-1]), dim1=1, dim2=2), dim=1)-self.board_size)*exponent)
        loss = r1+r2+r3+r4
        return loss


    def train_step(self, globalstep, cooperative=False):
        self.fixed_player_model.load_state_dict(self.player_model.state_dict())
        self.player_model.train()
        self.fixed_player_model.eval()
        self.player_model_old.eval()

        if cooperative: #train both players in one step
            self.zero_grad()
            total_loss1, _ = self.forward(0, global_step=globalstep)
            _, total_loss2 = self.forward(1, global_step=globalstep)
            loss = total_loss1+total_loss2
            loss.backward()
            self.optimizer1.step()
            self.scheduler1.step()
        else:
            #train player1
            total_loss1 = total_loss2=0
            for it in range(1):
                self.zero_grad()
                total_loss1, _ = self.forward(0, global_step=globalstep)
                loss = total_loss1
                loss.backward()
                self.optimizer1.step()
                self.scheduler1.step()

            self.fixed_player_model.load_state_dict(self.player_model.state_dict())
            #train player2
            for it in range(1):
                self.zero_grad()
                _, total_loss2 = self.forward(1, global_step=globalstep)
                loss = total_loss2
                loss.backward()
                self.optimizer1.step()
                self.scheduler1.step()

        return total_loss1+total_loss2

    def copy_weights(self):
        self.player_model_old.load_state_dict(self.player_model.state_dict())


    def model_average(self,model1, model2, alpha):
        # averages the wieights from both into model2
        # alpha is the weight of model1
        sdA = model1.state_dict()
        sdB = model2.state_dict()

        # Average all parameters
        for key in sdA:
            sdB[key] = sdB[key]*(1-alpha) + sdA[key]*alpha

        #load averaged state_dict (or use modelA/B)
        model2.load_state_dict(sdB)

    def average_weights(self, alpha=0.2):
        self.model_average(self.player_model,self.player_model_old, alpha)

    def save_to_pool(self):
        model = ConvSimple(4, self.feature_maps, 1, self.board_size).to(Config.device)
        model.load_state_dict(self.player_model.state_dict())
        self.model_pool.append(model)

    def predict_step(self, globalstep):
        self.eval()
        with torch.no_grad():
            total_loss1, total_loss2 = self.forward(0, rounding=False, training=False, global_step=globalstep)

        return total_loss1, total_loss2


