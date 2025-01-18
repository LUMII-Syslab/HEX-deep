import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from torch.optim.lr_scheduler import LambdaLR

from model.conv import ConvMove
from torch.utils.tensorboard import SummaryWriter
from utils.reinmax import reinmax
from config import Config
swap_allowed = Config.swap_allowed

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape, device=Config.device)
    return -torch.log(-torch.log(U+eps)+eps)

def softmax_global_soft(x):
    s = x.shape
    return torch.softmax(x.view([s[0],-1]), dim=-1).view(s)

def cross_entropy(logits, target, label_smoothing=0.1):
    soft_labels = target*(1-label_smoothing)+label_smoothing/2
    loss = torch.binary_cross_entropy_with_logits(logits, soft_labels)
    return loss.mean()

def softmax_global(x, remaining_probs, disable_occupied_positions=True):
    s = x.shape
    # probs = nn.functional.softplus(x)
    # probs = probs / (torch.sum(probs, dim=[2,3], keepdim=True) + 1e-6)
    # return probs
    #return torch.softmax(x.view([s[0],-1]), dim=-1).view(s)
    x_flat = x.view([s[0],-1])

    # a, b = reinmax(x_flat,1.5)
    # return a.view(s)

    #y_hard = torch.nn.functional.gumbel_softmax(x_flat, tau=1.0, hard=True, dim=-1)
    y_soft = torch.softmax(x_flat, dim=-1)
    if disable_occupied_positions:
        y_soft = y_soft * remaining_probs.view([s[0],-1])
    #y_soft = y_soft / (torch.sum(y_soft, dim=-1, keepdim=True)+1e-8)

    index = y_soft.max(-1, keepdim=True)[1]
    y_hard = torch.zeros_like(x_flat, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret.view(s)

def sabs(x):
    return torch.sqrt(torch.square(x)+1)-1
class HexGame(nn.Module):

    def __init__(self, feature_maps=256, rounds=15,board_size=6,batch_size=128, summary_writer=None,**kwargs):
    #def __init__(self, feature_maps=256, rounds=20, board_size=7, batch_size=128, summary_writer=None, **kwargs):
        super(HexGame, self).__init__(**kwargs)
        self.name = "Hex"
        self.rounds = rounds
        self.board_size = board_size
        self.feature_maps = feature_maps
        self.batch_size = batch_size
        self.move_layers = 3
        self.regul_weight = 0.001 # initially 0.001
        self.player_model = self.new_model()
        self.fixed_player_model = self.new_model()
        self.player_model_old = self.new_model()
        self.model_pool = []

        def warmup(current_step: int):
            warmup_steps = 100
            if current_step < warmup_steps:  # current_step / warmup_steps * base_lr
                return float(current_step / warmup_steps)
            else:
                return 1.

        self.optimizer1 = torch.optim.AdamW(self.player_model.parameters(), lr=Config.learning_rate, betas=(0.5, 0.999))
        self.scheduler1 = LambdaLR(self.optimizer1, lr_lambda=warmup)
        self.path_graph = self.create_graph()
        self.summary_writer = summary_writer

    def new_model(self):
        return ConvMove(5, self.feature_maps, 1, self.board_size).to(Config.device)

    def transposed_call(self, model, inputs, state):
        move_logits, state, paths = model(torch.transpose(inputs, 2,3), state)
        # note that we do not transpose state as it is kept transposed all the times
        return torch.transpose(move_logits,2,3).contiguous(), state, torch.transpose(paths,2,3).contiguous()

    def forward(self, train_player_id, player1_random_prob = 0.0, rounding=False, training=True, global_step=0):
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
            if random.random() < 0.5 and training:
                model2 = self.player_model_old
                #model2 = random.choice(self.model_pool)
        else:
            model1 = self.fixed_player_model
            model2 = self.player_model
            if random.random() < 0.5 and training:
                model1 = self.player_model_old
                #model1 = random.choice(self.model_pool)

        state1 = model1.initial_state(self.batch_size)
        state2 = torch.transpose(model2.initial_state(self.batch_size), 2,3)

       # probfunc = lambda x: torch.square(x) + 0.1 * x/self.board_size
        path_round = random.randrange(self.rounds)
        player1_looses = torch.tensor(1.0)
        player2_looses = torch.tensor(1.0)
        our_win_reward_lin = 0.0
        op_win_reward_lin = 0.0

        for step in range(self.rounds):
            noise_scale = 1.0# + 3.0 * torch.square(torch.rand((self.batch_size, 1, 1, 1),device=Config.device))
            mul_noise_scale = 0.0
            op_noise_scale = 0.0#6.0 * torch.square(torch.rand((self.batch_size, 1, 1, 1),device=Config.device))

            move1_logits,state1, paths1 = model1(torch.cat([player1_probs, player2_probs,1-player1_probs-player2_probs, move2, torch.zeros_like(player1_probs)], dim=1), state1)
            regul_loss1 = self.calc_regul(move1_logits, player1_probs + player2_probs)
            if train_player_id==1:
                move1_logits = move1_logits.detach()
                state1 = state1.detach()
            #logits_norm1 = torch.mean(torch.abs(move1_logits), dim=[1,2,3], keepdim=True)
            logits_norm1 = torch.std(move1_logits, dim=[1, 2, 3],correction=0, keepdim=True)
            move1_logits_org = move1_logits
            noise_scale1 = mul_noise_scale*logits_norm1+noise_scale+op_noise_scale*train_player_id #use more noise for the opponent

            if rounding:
                move1 = self.discretize(move1_logits)
            else:
                if training:move1_logits = move1_logits+sample_gumbel(move1_logits.shape) * noise_scale1
                move1 = softmax_global(move1_logits, 1-player1_probs-player2_probs, disable_occupied_positions = not training)

            player1_probs_prev = player1_probs

            if random.random() < player1_random_prob and training:
               move1, player1_probs = self.random_move(player1_probs, player2_probs)
            else:
                move1,player1_probs = self.move(move1, player1_probs, player2_probs)
            move_order = move_order + move1*(2*step)
            our_win_reward = torch.tensor(0.)
            if step == self.rounds - 1:
                #probfunc = torch.square# if train_player_id==0 else lambda x:x
                #probfunc = lambda x: x
                reverse_probs = train_player_id==1
                our_win_reward, path_map, player1_looses, our_win_reward_lin = self.win_loss(player1_probs, training, 0, reverse_probs=reverse_probs, move_order=move_order)
                # if train_player_id == 1:
                #     our_win_reward = our_win_reward*(1-player1_looses)
                # if train_player_id == 0:
                #     our_win_reward += 0.1*our_reward_lin*(1-player1_looses)
            else:
                path_map = softmax_global_soft(move1_logits_org)[:,0,:,:].detach().cpu().numpy()

            #move2_logits,state2 = model2(torch.cat([player2_probs, player1_probs,1-player1_probs-player2_probs, move1], dim=1),state2)
            move2_logits, state2, paths2 = self.transposed_call(model2,torch.cat([player2_probs, player1_probs, 1 - player1_probs - player2_probs, move1, torch.zeros_like(player1_probs)+(1 if swap_allowed and step==0 else 0)], dim=1), state2)
            regul_loss2 = self.calc_regul(move2_logits, (player1_probs + player2_probs)*(0 if swap_allowed and step==0 else 1))

            if train_player_id==0:
                move2_logits = move2_logits.detach()
                state2 = state2.detach()

            #logits_norm2 = torch.mean(torch.abs(move2_logits), dim=[1,2,3], keepdim=True)
            logits_norm2 = torch.std(move2_logits, dim=[1, 2, 3], correction=0, keepdim=True)
            move2_logits_org = move2_logits
            noise_scale2 = mul_noise_scale*logits_norm2 + noise_scale + op_noise_scale * (1-train_player_id)  # use more noise for the opponent

            if rounding:
                move2 = self.discretize(move2_logits)
            else:
                if training: move2_logits = move2_logits+sample_gumbel(move2_logits.shape) * noise_scale2
                move2 = softmax_global(move2_logits,1 - player1_probs - player2_probs, disable_occupied_positions = not (training or (swap_allowed and step==0)))

            player2_probs_prev = player2_probs

            if random.random() < player1_random_prob and training:
                move2, player2_probs = self.random_move(player2_probs, player1_probs)
            else:
                if step==0 and swap_allowed:
                    move2, player2_probs, player1_probs = self.swap_move(move2, player2_probs, player1_probs)
                else:
                    move2,player2_probs = self.move(move2, player2_probs, player1_probs)
            move_order = move_order + move2 * (2 * step+1)
            op_win_reward=torch.tensor(0.)
            if step==self.rounds-1:
                #probfunc = torch.square# if train_player_id == 1 else lambda x: x
                #probfunc = lambda x: x
                reverse_probs = train_player_id == 0
                op_win_reward, op_path_map, player2_looses, op_win_reward_lin = self.win_loss(player2_probs, training, 1, reverse_probs=reverse_probs, move_order=move_order)
                # if train_player_id == 0:
                #     op_win_reward = op_win_reward*(1-player2_looses)
                # if train_player_id == 1:
                #     op_win_reward += 0.1*op_win_reward_lin*(1-player2_looses)
            else:
                op_path_map=softmax_global_soft(move2_logits_org)[:,0,:,:].detach().cpu().numpy()

            # regul_loss1 = torch.var(move1_logits_org, correction=0)
            # regul_loss2 = torch.var(move2_logits_org, correction=0)
            opponont_weight = 1.0#1/self.board_size
            total_loss1 = (total_loss1+torch.mean(our_win_reward) +
                           opponont_weight*torch.mean(op_win_reward*(1-player2_looses)) +
                           regul_loss1*self.regul_weight/(step+1))
            total_loss2 = (total_loss2+torch.mean(op_win_reward) +
                           opponont_weight*torch.mean(our_win_reward*(1-player1_looses)) +
                           regul_loss2*self.regul_weight/(step+1))

            if train_player_id == 0:
                total_loss1 += torch.mean(0.1 * our_win_reward_lin * (1 - player1_looses))
            if train_player_id == 1:
                total_loss2 += torch.mean(0.1 * op_win_reward_lin * (1 - player2_looses))

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
                self.summary_writer.add_histogram('move/' + str(step * 2), move1_logits_org[0:1, ...], global_step=global_step)
                self.summary_writer.add_histogram('move/' + str(step * 2 + 1), move2_logits_org[0:1, ...], global_step=global_step)

            if step == path_round: #in one of the steps predict paths
                if train_player_id==0:
                    st_connected = self.get_connection_maps(player2_probs_prev,0)
                    path_loss = cross_entropy(paths1, st_connected)
                    total_loss1 += path_loss*0.1
                    paths_prediction = torch.sigmoid(paths1)
                else:
                    st_connected = self.get_connection_maps(player1_probs,1)
                    path_loss = cross_entropy(paths2, st_connected)
                    total_loss2 += path_loss*0.1
                    paths_prediction = torch.sigmoid(paths2)

        if global_step % 20 == 1:
            self.show_image(st_connected[:,0:1,:,:], st_connected[:,1:2,:,:], val_str + 'paths/a', None, global_step, probfunc=lambda x:x)
            self.show_image(paths_prediction[:, 0:1, :, :], paths_prediction[:, 1:2, :, :], val_str + 'paths/b', None, global_step, probfunc=lambda x:x)
            self.summary_writer.add_scalar("loss/path" + val_str, path_loss, global_step)
            #self.summary_writer.add_scalar("loss/all"+val_str,total_loss1+total_loss2, global_step)
            self.summary_writer.add_scalar("loss/green"+val_str, total_loss1, global_step)
            self.summary_writer.add_scalar("loss/red"+val_str, total_loss2, global_step)
            self.summary_writer.add_scalar("regul", regul_loss, global_step)
            self.summary_writer.add_histogram("last_state", state1[0:10,...], global_step)
            self.summary_writer.add_scalar("win/player1"+val_str, (1-player1_looses).mean(), global_step)
            self.summary_writer.add_scalar("win/player2"+val_str, (1-player2_looses).mean(), global_step)
            # self.summary_writer.add_histogram("path/logits", paths2[0:10, ...], global_step)
            # self.summary_writer.add_histogram("path/pred", paths_prediction[0:10, ...], global_step)


        return total_loss1, total_loss2

    def calc_regul(self,logits,occupied):
        # logits = logits - torch.mean(logits, dim=[1,2,3], keepdim=True)
        # regul_loss = torch.mean(torch.square(logits)*(1-occupied).detach(), dim=[1,2,3])
        logits = logits.view([logits.shape[0],-1])
        log_probs = torch.log_softmax(logits,dim=-1)
        free = 1.01-occupied.view([occupied.shape[0],-1])
        free_prob_sum = torch.sum(free, dim=-1, keepdim=True)
        target_probs = free/free_prob_sum
        #target_probs = torch.zeros_like(free)+1/(self.board_size**2)
        kl = torch.kl_div(log_probs, target_probs)
        return kl.sum(dim=-1).mean()

    def show_image(self, probs,other_probs, name, path_mask, globalstep, probfunc = torch.sqrt):
        image = probfunc(probs[0,:,:,:]).detach().cpu().numpy()
        image1 = probfunc(other_probs[0,:,:,:]).detach().cpu().numpy()
        if path_mask is not None:
            path_mask = np.sqrt(path_mask[0:1,:,:])
        else:
            path_mask = np.zeros([1,self.board_size, self.board_size])

        image = np.concatenate([image1, image,path_mask], axis=0)
        image = np.repeat(image, 2, axis=2)

        image = np.pad(image, [[0,0],[0,0],[0,self.board_size-1]],'constant', constant_values=0.5)
        # shear x
        for i in range(image.shape[1]):
            image[:,i, :] = np.roll(image[:,i, :], i)
        # image = tfa.image.shear_x(tf.cast(image[0,:,:,:]*255, tf.uint8), -1., 128)
        image = np.repeat(image, 2, axis=1)
        image = (image*255).astype(np.uint8)
        self.summary_writer.add_image(name, image, globalstep)

    def move(self, move_probs, our_probs, opponent_probs, normalize=False):
        remaining_probs = 1-(our_probs+opponent_probs)
        move_probs = move_probs*remaining_probs
        if normalize:
            move_probs = move_probs / (torch.sum(move_probs, dim=[1,2,3], keepdim=True) + 1e-6)
            move_probs = torch.minimum(move_probs, remaining_probs)
        return move_probs, our_probs+move_probs

    # move when a stone is allowed to replace the opponents stone
    # It is assumed that this is move #1
    def swap_move(self, move_probs, our_probs, opponent_probs):
        overlapped_move = move_probs*opponent_probs
        opponent_probs = opponent_probs*(1 - move_probs) #opponent stone is removed if we put in the same place
        move_probs = move_probs - overlapped_move + torch.transpose(overlapped_move, 2, 3) #our move is transposed if replacement occurs
        return move_probs, our_probs+move_probs, opponent_probs

    def discretize(self, move):
        randomized_probs = move.view([self.batch_size, self.board_size * self.board_size])
        move_index = torch.argmax(randomized_probs, dim=-1)  # choose one of the empty positions
        move_mask = torch.nn.functional.one_hot(move_index, self.board_size * self.board_size)
        move_mask = move_mask.view([self.batch_size, 1,self.board_size, self.board_size])
        return move_mask

    def random_move(self, player1_probs, player2_probs):
        move_mask_all = torch.zeros([self.batch_size, 1,self.board_size,self.board_size], device = player1_probs.device)
        # choose random move
        for trials in range(5):
            player_probs = player1_probs + player2_probs
            randomized_probs = 1-player_probs + torch.rand(player_probs.shape, device = player_probs.device)
            move_mask = self.discretize(randomized_probs)
            move_sum = torch.sum(move_mask_all, dim=[2, 3], keepdim=True)
            move_mask = move_mask*(1 - move_sum)
            move_mask, player1_probs = self.move(move_mask, player1_probs, player2_probs, normalize=False)
            move_mask_all = move_mask_all+move_mask
            move_sum = torch.sum(move_mask_all)
            if move_sum >= self.batch_size * 0.95: break

        return move_mask_all, player1_probs

    def create_graph(self):
        G = nx.DiGraph()
        #left_node = G.add_node((-1,-1))
        #right_node = G.add_node((-2,-2))

        for y in range(self.board_size):
            G.add_edge((-1,-1),(0,y))
            G.add_edge((0, y), (-1, -1))
            G.add_edge((self.board_size-1, y),(-2,-2))
            G.add_edge((-2, -2), (self.board_size - 1, y))
            for x in range(self.board_size):
                if x+1<self.board_size:
                    G.add_edge((x,y), (x+1,y))
                    G.add_edge((x+1, y), (x, y))
                if y + 1 < self.board_size:
                    G.add_edge((x, y), (x, y+1))
                    G.add_edge((x, y+1), (x, y))
                if x + 1 < self.board_size and y-1 >= 0:
                    G.add_edge((x, y), (x+1, y - 1))
                    G.add_edge((x+1, y-1), (x, y))
        return G

    def win_loss(self, probs, training, player_index=0, prob_func = torch.square, reverse_probs=False, move_order=None):
        probs = 1-probs[:,0,:,:]
        probs_org = probs
        probs_noisy = probs
        if training:
            #move_order = move_order + torch.abs(torch.randn_like(probs)) * self.board_size * 2
            #order_weight = 0.005 / self.board_size
            #probs_noisy = probs_noisy * (1 + move_order[:, 0, :, :] * order_weight)  # give higher probability to earlier moves of the opponent
            probs_noisy = nn.functional.softplus(probs_noisy*6-3+torch.randn(probs.shape, device=Config.device)*1.0)
            #probs_noisy = torch.relu(probs_noisy+(torch.randn_like(probs)+1)*order_weight*0.2)
            #probs_noisy = probs * tf.exp(tf.random.normal(tf.shape(probs), stddev=0.3))
            #probs_noisy += tf.random.uniform(tf.shape(probs), maxval = 0.03)
        probs_np = torch.square(probs_noisy.detach()).cpu().numpy()
        if reverse_probs:
            probs = 1-probs
        probs_sq = prob_func(probs)
        player_index_np = player_index#.numpy()

        # calculate if path exists from left to right using the current weights
        path_mask = np.zeros([self.batch_size, self.board_size, self.board_size], dtype=np.float32)

        for batch_id in range(self.batch_size):
            def weight_func(n1,n2,attr_map):
                if n1[0] < 0: return 0 # zero weight from source node
                if player_index_np==0:
                    weight = probs_np[batch_id, n1[0],n1[1]]
                else:
                    weight = probs_np[batch_id, n1[1], n1[0]]
                return weight

            path = nx.dijkstra_path(self.path_graph, (-1,-1), (-2,-2), weight=weight_func)
            #single_source_dijkstra_path_length
            assert len(path) >= self.board_size + 2

            for node in path:
                if node[0]>=0:
                    if player_index_np == 0:
                        path_mask[batch_id, node[0], node[1]] = 1.
                    else:
                        path_mask[batch_id, node[1], node[0]] = 1.


        #log_path_len = tf.reduce_sum(probs*path_mask, axis=[1,2])
        path_min = torch.amax(probs_org * torch.tensor(path_mask, device=Config.device), dim=[1, 2]) #0 if player wins, 1 if looses
        path_sum_sq = torch.sum(probs_sq.mul(torch.tensor(path_mask, device=Config.device)), dim=[1, 2])
        path_sum_lin = torch.sum(probs.mul(torch.tensor(path_mask, device=Config.device)), dim=[1, 2])
        #loss = tf.reduce_mean(path_min+0.1*path_sum)
        #example_loss = torch.mean(path_sum)
        return path_sum_sq, path_mask, path_min, path_sum_lin

    def get_connection_maps(self, probs, player_index=0):
        probs = 1-probs[:,0,:,:]
        probs_np = probs.detach().cpu().numpy()

        # calculate if path exists from left to right using the current weights
        path_mask = np.zeros([self.batch_size, 2, self.board_size, self.board_size], dtype=np.float32)

        for batch_id in range(self.batch_size):
            def weight_func_forward(n1, n2, attr_map):
                if n2[0] < 0: return 100  #large weight for target
                if player_index == 0:
                    weight = probs_np[batch_id, n2[0], n2[1]]
                else:
                    weight = probs_np[batch_id, n2[1], n2[0]]
                return weight

            source_dist = nx.single_source_dijkstra_path_length(self.path_graph, (-1,-1), cutoff=0.99, weight=weight_func_forward)
            target_dist = nx.single_source_dijkstra_path_length(self.path_graph, (-2, -2), cutoff=0.99, weight=weight_func_forward)

            for node in source_dist:
                if node[0]>=0:
                    if player_index == 0:
                        path_mask[batch_id, 0, node[0], node[1]] = 1.
                    else:
                        path_mask[batch_id, 0, node[1], node[0]] = 1.

            for node in target_dist:
                if node[0]>=0:
                    if player_index == 0:
                        path_mask[batch_id, 1, node[0], node[1]] = 1.
                    else:
                        path_mask[batch_id, 1, node[1], node[0]] = 1.

        return torch.tensor(path_mask, device=Config.device)


    def train_step(self, globalstep, cooperative=False):
        self.fixed_player_model.load_state_dict(self.player_model.state_dict())
        self.player_model.train()
        self.fixed_player_model.eval()
        self.player_model_old.eval()

        if cooperative: #train both players in one step
            pass
            # with tf.GradientTape() as tape:
            #     total_loss1, total_loss2 = self.call(0., training=True)
            #     total_loss1a, total_loss2a = self.call(1., training=True)
            #     loss = total_loss1+total_loss2a#+total_loss1a+total_loss2
            #
            # vars = self.trainable_variables
            # gradients = tape.gradient(loss, vars,unconnected_gradients='zero')
            # self.optimize(gradients, vars)
        else:
            #train player1
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
        model = self.new_model()
        model.load_state_dict(self.player_model.state_dict())
        self.model_pool.append(model)

    def predict_step(self, globalstep):
        self.eval()
        with torch.no_grad():
            total_loss1, total_loss2 = self.forward(0, rounding=True, training=False, global_step=globalstep)

        return total_loss1, total_loss2


