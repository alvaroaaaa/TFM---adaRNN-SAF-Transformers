import torch.nn as nn
import torch
import torch.optim as optim

import os
import argparse
import datetime
import numpy as np

from utils import utils
from base.AdaTransformer import AdaTransformer

import pretty_errors
import dataset.data_process as data_process
import matplotlib.pyplot as plt
from itertools import product


def get_index(num_domain):
    index = []
    for i in range(num_domain):
        for j in range(i+1, num_domain+1):
            index.append((i, j))
    return index

def train_epoch(args, model, optimizer, backcast_optimizer, train_loader_list, epoch, dist_old=None, weight_mat=None):

    model.train()
    criterion = nn.MSELoss()
    loss_all = []
    loss_1_all = []
    dist_mat = torch.zeros(args.num_layer, args.len_seq).cuda()
    len_loader = np.inf
    
    for data_all in zip(*train_loader_list):
        optimizer.zero_grad()
        list_feat = []
        list_label = []
        for data in data_all:
            feature, label_reg = data[0].cuda().float(), data[2].cuda().float()
            list_feat.append(feature)
            list_label.append(label_reg)
        flag = False
        index = get_index(len(data_all) - 1)
        for temp_index in index:
            s1, s2 = temp_index
            if list_feat[s1].shape[0] != list_feat[s2].shape[0]:
                flag = True
                break
        if flag:
            continue
        total_loss = torch.zeros(1).cuda()
    
        for i in range(len(index)):
            feature_s = list_feat[index[i][0]]
            feature_t = list_feat[index[i][1]]
            label_reg_s = list_label[index[i][0]]
            label_reg_t = list_label[index[i][1]]
            feature_all = torch.cat((feature_s, feature_t), 0)

            backcast_loss = model.encode_and_backcast(feature_all)
            backcast_optimizer.zero_grad()
            backcast_loss.backward(retain_graph=True) 
            backcast_optimizer.step()

            if epoch < args.pre_epoch:
                pred_all, loss_adapt, out_weight_list = model.forward_pre_train(feature_all, args.loss_type, len_win=args.len_win)

            else:
                pred_all, loss_adapt, dist, weight_mat = model.forward_Boosting(feature_all, weight_mat)
                dist_mat = dist_mat + dist
        
            pred_s = pred_all[0:feature_s.size(0)]
            pred_t = pred_all[feature_s.size(0):]
            pred_s = torch.mean(pred_s, dim=1).view(pred_s.shape[0])
            pred_t = torch.mean(pred_t, dim=1).view(pred_t.shape[0])

            loss_s = criterion(pred_s, label_reg_s)
            loss_t = criterion(pred_t, label_reg_t)

            total_loss += loss_s + args.dw * loss_adapt + loss_t

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()

    loss = np.array(loss_all).mean(axis=0)
    loss_l1 = np.array(loss_1_all).mean()
    if epoch >= args.pre_epoch:
        if epoch > args.pre_epoch:
            weight_mat = model.update_weight_Boosting(weight_mat, dist_old, dist_mat)
        return loss, loss_l1, weight_mat, dist_mat
    else:
        weight_mat = transform_type(out_weight_list, args)
        return loss, loss_l1, weight_mat, None

def transform_type(init_weight, args):
    num_layer = args.num_layer
    len_seq = args.len_seq
    weight = torch.ones(num_layer, len_seq).cuda()
    for i in range(num_layer):
        for j in range(len_seq):
            weight[i, j] = init_weight[i][j].item()
    return weight

def test_epoch(model, test_loader, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    for feature, _, label_reg in test_loader:
        feature, label_reg = feature.cuda().float(), label_reg.cuda().float()
        with torch.no_grad():
            pred = model.predict(feature)
            pred = torch.mean(pred,dim=1).view(pred.shape[0])
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = loss_r / len(test_loader)
    return loss, loss_1, loss_r


def test_epoch_inference(model, test_loader, prefix='Test'):
    model.eval()
    total_loss = 0
    total_loss_1 = 0
    total_loss_r = 0
    criterion = nn.MSELoss()
    criterion_1 = nn.L1Loss()
    i = 0
    for feature, _, label_reg in test_loader:
        feature, label_reg = feature.cuda().float(), label_reg.cuda().float()
        with torch.no_grad():
            pred = model.predict(feature)
            pred = torch.mean(pred,dim=1).view(pred.shape[0])
        loss = criterion(pred, label_reg)
        loss_r = torch.sqrt(loss)
        loss_1 = criterion_1(pred, label_reg)
        total_loss += loss.item()
        total_loss_1 += loss_1.item()
        total_loss_r += loss_r.item()
        if i == 0:
            label_list = label_reg.cpu().numpy()
            predict_list = pred.cpu().numpy()
        else:
            label_list = np.hstack((label_list, label_reg.cpu().numpy()))
            predict_list = np.hstack((predict_list, pred.cpu().numpy()))

        i = i + 1
    loss = total_loss / len(test_loader)
    loss_1 = total_loss_1 / len(test_loader)
    loss_r = total_loss_r / len(test_loader)
    return loss, loss_1, loss_r, label_list, predict_list


def inference(model, data_loader):
    loss, loss_1, loss_r, label_list, predict_list = test_epoch_inference(
        model, data_loader, prefix='Inference')
    return loss, loss_1, loss_r, label_list, predict_list


def inference_all(output_path, model, model_path, loaders):
    loss_list = []
    loss_l1_list = []
    loss_r_list = []
    model.load_state_dict(torch.load(model_path))
    i = 0
    list_name = ['train', 'valid', 'test']
    for loader in loaders:
        loss, loss_1, loss_r, label_list, predict_list = inference(
            model, loader)
        loss_list.append(loss)
        loss_l1_list.append(loss_1)
        loss_r_list.append(loss_r)
        i = i + 1
    return loss_list, loss_l1_list, loss_r_list


def main_transfer(args):
    print(args)

    output_path = args.outdir + '_' + args.station + '_' + args.model_name + '_weather_' + \
        args.loss_type + '_' + str(args.pre_epoch) + \
        '_'  + '_' + str(args.lr) + "_" + str(args.train_type) + "-layer-num-" + str(args.num_layer) + "-hidden-" + str(args.hidden_dim) + "-num_head-" + str(args.num_head) + "dw-" + str(args.dw)
    save_model_name = args.model_name + '_' + args.loss_type + \
        '_' + str(args.dw) + '_' + str(args.lr) + '.pkl'
    utils.dir_exist(output_path)

    train_loader_list, valid_loader, test_loader = data_process.load_weather_data_multi_domain("./dataset", args.batch_size, args.station, 2)

    args.log_file = os.path.join(output_path, 'run.log')
    ######
    # Model parameters
    d_model = args.hidden_dim #32  Lattent dim
    h = args.num_head #4   Number of heads
    N = args.num_layer  # Number of encoder and decoder to stack
    d_input = 6  # From dataset
    d_output = 1  # From dataset
    pe_period = args.len_seq
    dropout = args.dropout

    model = AdaTransformer(d_input, d_model, d_output, h, N, dropout = dropout, pe_period=pe_period).cuda()


    optimizer = torch.optim.Adam(
        list(model._embedding.parameters()) +
        list(model.layers_encoding.parameters()) +
        list(model.layers_decoding.parameters()) +
        list(model.bottleneck.parameters()) +
        list(model.fc.parameters()) +
        [param for gate in model.gate for param in gate.parameters()] +  # Parameters from gate weights
        [param for bn in model.bn_lst for param in bn.parameters()],  # Parameters from batch normalization layers
        lr=args.lr)

    backcast_optimizer = torch.optim.Adam(
        list(model.layers_encoding.parameters()) +
        list(model.layers_backasting.parameters()),  # Include only backcast decoder parameters
        lr=0.001)
   
    best_score = np.inf
    best_epoch, stop_round = 0, 0
    weight_mat, dist_mat = None, None

    for epoch in range(args.n_epochs):

        loss, loss1, weight_mat, dist_mat = train_epoch(
                args, model, optimizer, backcast_optimizer, train_loader_list, epoch, dist_mat, weight_mat)

        print('evaluating...')
        train_loss, train_loss_l1, train_loss_r = test_epoch(
            model, train_loader_list[0], prefix='Train')
        val_loss, val_loss_l1, val_loss_r = test_epoch(
            model, valid_loader, prefix='Valid')
        test_loss, test_loss_l1, test_loss_r = test_epoch(
            model, test_loader, prefix='Test')

        print('train %.6f, valid %.6f, test %.6f' %
               (train_loss, val_loss, test_loss))
        print('train %.6f, valid %.6f, test %.6f' %
               (train_loss_l1, val_loss_l1, test_loss_l1))
        if val_loss < best_score:
            best_score = val_loss
            stop_round = 0
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                output_path, save_model_name))
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                print('early stop')
                break

    print('best val score:', best_score, '@', best_epoch)

    loaders = train_loader_list[0], valid_loader, test_loader
    loss_list, loss_l1_list, loss_r_list = inference_all(output_path, model, os.path.join(
        output_path, save_model_name), loaders)
    print('MSE: train %.6f, valid %.6f, test %.6f' %
           (loss_list[0], loss_list[1], loss_list[2]))
    print('L1:  train %.6f, valid %.6f, test %.6f' %
           (loss_l1_list[0], loss_l1_list[1], loss_l1_list[2]))
    print('RMSE: train %.6f, valid %.6f, test %.6f' %
           (loss_r_list[0], loss_r_list[1], loss_r_list[2]))
    print('Finished.')

    return loss_r_list[2]


def get_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='adapt_tf')
    parser.add_argument('--d_feat', type=int, default=2)

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--class_num', type=int, default=1)
    parser.add_argument('--pre_epoch', type=int, default=10)  # 25
    parser.add_argument('--num_layer', type=int, default=2)  # 25

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--dw', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='cosine')
    parser.add_argument('--train_type', type=str, default='all')
    parser.add_argument('--station', type=str, default='Nongzhanguan')
    parser.add_argument('--data_mode', type=str,
                        default='pre_process')
    parser.add_argument('--num_domain', type=int, default=2)
    parser.add_argument('--len_seq', type=int, default=24)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_head', type=int, default=8)

    # other
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--data_path', default="./dataset")
    parser.add_argument('--outdir', default='./outputs')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--log_file', type=str, default='run.log')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--len_win', type=int, default=0)
    args = parser.parse_args()

    return args

def run_parameter_tuning():
    # Define lists of values for each tunable parameter
    dropout_values = [0.1, 0.2]
    pre_epoch_values = [0, 10, 20, 50]
    num_layers_values = [1, 2, 4]
    n_epochs_values = [100, 200]
    lr_values = [1e-3, 5e-4]
    dw_values = [0.3, 0.5, 0.7]
    len_seq_values = [24]
    hidden_dim_values = [32, 64, 128]
    num_head_values = [2, 4, 8]

    # Generate all combinations of parameter values
    all_combinations = list(product(dropout_values, pre_epoch_values, num_layers_values, n_epochs_values, lr_values, dw_values, len_seq_values, hidden_dim_values, num_head_values))

    best_performance = float('inf')  # Adjust based on whether you're minimizing or maximizing
    best_combination = None

    for combination in all_combinations:
        args = get_args()  # Get the default args
        # Unpack the combination and set the parameters
        args.dropout, args.pre_epoch, args.num_layers, args.n_epochs, args.lr, args.dw, args.len_seq, args.hidden_dim, args.num_head = combination
        
        performance_metric = main_transfer(args)  # Run training and get performance metric
        
        # Update the best combination if the current one performs better
        if performance_metric < best_performance:  # Adjust comparison for your metric
            best_performance = performance_metric
            best_combination = combination

    print(f"Best parameter combination: {best_combination}")
    print(f"Best performance: {best_performance}")

def modify_args_for_combination(args, combination):
    """
    Modifies the args based on the given parameter combination.
    """
    for key, value in combination.items():
        setattr(args, key, value)
    return args

if __name__ == '__main__':

    np.random.seed(69)
    torch.manual_seed(69)
    torch.cuda.manual_seed_all(69)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    run_parameter_tuning()