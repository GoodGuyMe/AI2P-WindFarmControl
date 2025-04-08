import json
import os
import torch

import numpy as np
import torch.nn as nn

from datetime import datetime

from torch.optim import Adam

from PIGNN.architecture.pignn import FlowPIGNN
from PIGNN.architecture.deconv import FCDeConvNet, DeConvNet
from LSTM.architecture.windspeedLSTM import WindspeedLSTM, WindSpeedLSTMDeConv
from PIGNN.dataset import create_data_loaders, get_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator()
generator.manual_seed(42)

def compute_loss(batch, criterion, model):
    # Logic to handle different model types
    x, pos, edge_attr, glob, target = batch.x, batch.pos, batch.edge_attr.float(), batch.global_feats.float(), batch.y
    # Concatenate features for non-GNN models
    if isinstance(model, FCDeConvNet):
        x_cat = torch.cat([x.flatten(), pos.flatten(), edge_attr.flatten(), glob.flatten()], dim=-1).float()
        pred = model(x_cat)
    else:
        nf = torch.cat((x, pos), dim=-1).float()
        pred = model(batch, nf, edge_attr, glob)
    loss = criterion(pred, target.reshape((pred.size(0), -1)))
    return loss


def train_epoch(train_loader, model, criterion, optimizer, scheduler):
    train_losses = []
    model.train()
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        loss = compute_loss(batch, criterion, model)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return np.mean(train_losses)


def eval_epoch(val_loader, model, criterion):
    val_losses = []
    with torch.no_grad():
        model.eval()
        for batch in val_loader:
            batch = batch.to(device)
            val_loss = compute_loss(batch, criterion, model)
            val_losses.append(val_loss.item())
    model.train()
    return np.mean(val_losses)


def train(model, train_params, train_loader, val_loader, output_folder):
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    train_loss_list = []
    val_loss_list = []

    num_epochs = train_params['num_epochs']
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Perform a training and validation epoch
        train_loss = train_epoch(train_loader, model, criterion, optimizer, scheduler)
        val_loss = eval_epoch(val_loader, model, criterion)
        learning_rate = optimizer.param_groups[0]['lr']
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {train_loss}, validation loss: {val_loss}")

        # Save model pointer
        torch.save(model.state_dict(), f"{output_folder}/pignn_{epoch}.pt")

        # Check early stopping criterion
        if epoch == num_epochs:
            np.save(f"{output_folder}/train_loss", train_loss_list)
            np.save(f"{output_folder}/val_loss", val_loss_list)
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{output_folder}/pignn_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params['early_stop_after']:
            np.save(f"{output_folder}/train_loss", train_loss_list)
            np.save(f"{output_folder}/val_loss", val_loss_list)
            print(f'Early stopping at epoch {epoch}')
            break


def process_temporal_batch(batch, graph_model, temporal_model, criterion, embedding_size, output_size):
    generated_img = []
    target_img = []
    for i, seq in enumerate(batch[0]):
        # Process graphs in parallel at each timestep for the entire batch
        seq = seq.to(device)
        nf = torch.cat((seq.x.to(device), seq.pos.to(device)), dim=-1).float()
        ef = seq.edge_attr.to(device).float()
        gf = seq.global_feats.to(device).float()
        graph_output = graph_model(seq, nf, ef, gf).reshape(-1, embedding_size[0], embedding_size[1])
        generated_img.append(graph_output)
        target_img.append(batch[1][i].y.to(device).reshape(-1, output_size[0], output_size[1]))

    temporal_img = torch.stack(generated_img, dim=1)
    output = temporal_model(temporal_img).flatten()
    target = torch.stack(target_img, dim=1).flatten()
    return criterion(output, target)


def train_temporal_epoch(train_loader, graph_model, temporal_model, criterion, optimizer, scheduler, embedding_size,
                         output_size):
    train_losses = []
    graph_model.train()
    temporal_model.train()
    for i, batch in enumerate(train_loader):
        if i < len(train_loader) - 1:
            loss = process_temporal_batch(batch, graph_model, temporal_model, criterion, embedding_size, output_size)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return np.mean(train_losses)


def eval_temporal_epoch(val_loader, graph_model, temporal_model, criterion, embedding_size, output_size):
    with torch.no_grad():
        graph_model.eval()
        temporal_model.eval()
        val_losses = []
        for i, batch in enumerate(val_loader):
            if i < len(val_loader) - 1:
                val_loss = process_temporal_batch(batch, graph_model, temporal_model, criterion, embedding_size,
                                                  output_size)
                val_losses.append(val_loss.item())
    graph_model.train()
    temporal_model.train()
    return np.mean(val_losses)


def train_temporal(graph_model, temporal_model, train_params, train_loader, val_loader, output_folder, embedding_size,
                   output_size):
    optimizer = Adam(list(graph_model.parameters()) + list(temporal_model.parameters()), lr=0.01)
    criterion = nn.MSELoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

    train_loss_list = []
    val_loss_list = []

    num_epochs = train_params['num_epochs']
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Perform a training and validation epoch
        train_loss = train_temporal_epoch(train_loader, graph_model, temporal_model, criterion, optimizer, scheduler,
                                          embedding_size, output_size)
        val_loss = eval_temporal_epoch(val_loader, graph_model, temporal_model, criterion, embedding_size, output_size)
        learning_rate = optimizer.param_groups[0]['lr']
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        print(f"step {epoch}/{num_epochs}, lr: {learning_rate}, training loss: {train_loss}, validation loss: {val_loss}")

        # Save model pointers
        torch.save(graph_model.state_dict(), f"{output_folder}/pignn_{epoch}.pt")
        torch.save(temporal_model.state_dict(), f"{output_folder}/unet_lstm_{epoch}.pt")

        if epoch == num_epochs:
            np.save(f"{output_folder}/train_loss", train_loss_list)
            np.save(f"{output_folder}/val_loss", val_loss_list)

        # Check early stopping criterion
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(graph_model.state_dict(), f"{output_folder}/pignn_best.pt")
            torch.save(temporal_model.state_dict(), f"{output_folder}/unet_lstm_best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= train_params['early_stop_after']:
            np.save(f"{output_folder}/train_loss", train_loss_list)
            np.save(f"{output_folder}/val_loss", val_loss_list)
            print(f'Early stopping at epoch {epoch}')
            break


def create_output_folder(train_config, net_type):
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    output_folder = f"results/{time}_Case0{train_config['case_nr']}_{train_config['wake_steering']}_{net_type}_" \
                    f"{train_config['seq_length']}"
    os.makedirs(output_folder)
    return output_folder


def save_config(output_folder, config):
    with open(f"{output_folder}/config.json", 'w') as f:
        json.dump(config, f)


def get_pignn_config():
    return {
        'edge_in_dim': 2,
        'node_in_dim': 3,
        'global_in_dim': 2,
        'n_pign_layers': 3,
        'edge_hidden_dim': 50,
        'node_hidden_dim': 50,
        'global_hidden_dim': 50,
        'num_nodes': 10,
        'residual': True,
        'input_norm': True,
        'pign_mlp_params': {
            'num_neurons': [256, 128],
            'hidden_act': 'ReLU',
            'out_act': 'ReLU'
        },
        'reg_mlp_params': {
            'num_neurons': [64, 128, 256],
            'hidden_act': 'ReLU',
            'out_act': 'ReLU'
        },
    }


def get_dataset_dirs(case_nr, wake_steering, max_angle, use_all_data):
    cases = [1, 2, 3] if use_all_data else [case_nr]
    wake_steering_cases = [True, False] if use_all_data else [wake_steering]
    folders = []

    for case in cases:
        for steering in wake_steering_cases:
            post_fix = "LuT2deg_internal" if steering else "BL"
            data_folder = f"../data/Case_0{case}/graphs/{post_fix}/{max_angle}"
            folders.append(data_folder)

    return folders


def get_config(case_nr=1, wake_steering=False, max_angle=30, use_graph=True, seq_length=1, batch_size=64,
               output_size=128, direct_lstm=False, num_epochs=300, early_stop_after=10, use_all_data=False):
    return get_pignn_config(), {
        'case_nr': case_nr,
        'wake_steering': wake_steering,
        'max_angle': max_angle,
        'num_epochs': num_epochs,
        'use_graph': use_graph,
        'early_stop_after': early_stop_after,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'direct_lstm': direct_lstm,
        'output_size': output_size,
        "dataset_dirs": get_dataset_dirs(case_nr, wake_steering, max_angle, use_all_data)
    }


def run(case_nr=1, wake_steering=False, max_angle=30, use_graph=True, seq_length=1, batch_size=64, direct_lstm=False,
        output_size=128, use_all_data=False):
    model_cfg, train_cfg = get_config(case_nr=case_nr, wake_steering=wake_steering, max_angle=max_angle,
                                      use_graph=use_graph, seq_length=seq_length, batch_size=batch_size,
                                      output_size=output_size, direct_lstm=direct_lstm, use_all_data=use_all_data)

    is_direct_lstm = train_cfg['direct_lstm']
    is_temporal = seq_length > 1
    pignn_type = ("pignn_lstm_deconv" if is_direct_lstm else "pignn_unet_lstm") if is_temporal else "pignn_deconv"
    net_type = f"{pignn_type}_{train_cfg['max_angle']}" if train_cfg['use_graph'] else "fcn_deconv"

    output_folder = create_output_folder(train_cfg, net_type)
    save_config(output_folder, train_cfg)

    dataset = get_dataset(train_cfg['dataset_dirs'], is_temporal, seq_length)
    train_loader, val_loader, test_loader = create_data_loaders(dataset, train_cfg['batch_size'], seq_length)

    out_size = (output_size, output_size)
    deconv_model = DeConvNet(1, [64, 128, 256, 1],
                            output_size=output_size) if not is_temporal or not is_direct_lstm else None
    graph_model = FlowPIGNN(**model_cfg, deconv_model=deconv_model).to(device) if \
        train_cfg['use_graph'] else FCDeConvNet(212, 650, 656, 500).to(device)

    print(graph_model)

    if is_temporal:
        temporal_model = WindSpeedLSTMDeConv(seq_length, [64, 128, 256, 1], output_size).to(
            device) if is_direct_lstm else WindspeedLSTM(seq_length).to(device)
        embedding_size = (50, 10) if is_direct_lstm else out_size
        train_temporal(graph_model, temporal_model, train_cfg, train_loader, val_loader, output_folder, embedding_size, out_size)
    else:
        train(graph_model, train_cfg, train_loader, val_loader, output_folder)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Run experiments with different configurations.')
    # parser.add_argument('--case_nr', type=int, default=1, help='Case number to use for the experiment (default: 1)')
    # parser.add_argument('--wake_steering', action='store_true', help='Enable wake steering (default: False)')
    # parser.add_argument('--max_angle', type=int, default=30, help='Maximum angle for the experiment (default: 30)')
    # parser.add_argument('--use_graph', action='store_true', help='Use graph representation (default: False)')
    # parser.add_argument('--seq_length', type=int, default=1, help='Sequence length for the experiment (default: 1)')
    # parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the experiment (default: 64)')
    # parser.add_argument('--direct_lstm', action='store_true', help='Feed the PIGNN output directly to the LSTM (default: False)')
    # parser.add_argument('--use_all_data', action='store_true', help='Use all available training data (default: False)')
    # args = parser.parse_args()
    #
    # run(case_nr=args.case_nr, wake_steering=args.wake_steering, max_angle=args.max_angle, use_graph=args.use_graph,
    # seq_length=args.seq_length, batch_size=args.batch_size, direct_lstm=args.direct_lstm, use_all_data=args.use_all_data)

    # run(case_nr=1, wake_steering=False, max_angle=30, seq_length=1, output_size=128)
    # run(case_nr=1, wake_steering=False, max_angle=90, seq_length=1, output_size=128)
    # run(case_nr=1, wake_steering=False, max_angle=360, seq_length=1, output_size=128)

    run(case_nr=1, wake_steering=True, max_angle=30, seq_length=50, batch_size=4, output_size=128)
    # run(case_nr=1, wake_steering=False, max_angle=30, seq_length=50, batch_size=4, output_size=128)
    # run(case_nr=1, wake_steering=True, max_angle=30, seq_length=50, batch_size=4, output_size=128, direct_lstm=True)
    # run(case_nr=1, wake_steering=True, max_angle=90, seq_length=1, output_size=128)
    # run(case_nr=1, wake_steering=True, max_angle=360, seq_length=1, output_size=128)
