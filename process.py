import os 
import sys 
import numpy as np 
from tqdm import tqdm 
from collections import OrderedDict 

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from data_feed import DataFeed 
from data_feed_test import DataFeed_Test
from model import CsinetPlus 
from utils import save_model, load_model, cal_nmse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(config, device):
    # Build the model
    model = CsinetPlus(config.enc_dim)

    # Setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if config.load_model_path:
        model = load_model(model, config.load_model_path)
    model = model.to(device)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100], gamma=0.1
    )

    return model, optimizer, scheduler

def train_process(config, seed=0):
    setup_seed(seed)
    device = torch.device(f'cuda:{config.gpu}')

    # Get output directory ready
    if not os.path.isdir(config.store_model_path):
        os.makedirs(config.store_model_path)

    # Create a summary writer with the specified folder name
    writer = SummaryWriter(os.path.join(config.store_model_path, 'summary'))

    # Prepare training data
    train_loader = DataLoader(
        DataFeed(config.data_root, config.train_csv, num_data_point=config.num_train_data),
        batch_size=config.batch_size,
        shuffle=True
    )

    val_feed = DataFeed(config.test_data_root, config.test_csv)
    val_loader = DataLoader(val_feed, batch_size=1024)

    # Build model
    model, optimizer, scheduler = build_model(config, device)
    print("Finish building model")

    # Define loss function
    loss_function = nn.MSELoss()

    # Training
    all_val_nmse = []
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        running_nmse = 0.0
        with tqdm(train_loader, unit="batch", file=sys.stdout) as tepoch:
            for i, data in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")

                # Get the inputs
                input_channel, data_idx = data[0].to(device), data[1].to(device)

                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                encoded_vector, output_channel = model(input_channel)
                loss = loss_function(output_channel, input_channel)

                nmse = torch.mean(cal_nmse(input_channel, output_channel), 0).item()

                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_nmse = (nmse + i * running_nmse) / (i + 1)
                log = OrderedDict()
                log["loss"] = "{:.6e}".format(running_loss)
                log["nmse"] = running_nmse
                tepoch.set_postfix(log)
            scheduler.step()
        
        # Validation
        val_loss = 0
        val_nmse = 0
        if epoch >= config.num_epochs - 50:
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    # Get the inputs
                    input_channel, data_idx = data[0].to(device), data[1].to(device)

                    # Forward
                    encoded_vector, output_channel = model(input_channel)
                    loss = loss_function(output_channel, input_channel)
                    nmse = torch.mean(cal_nmse(input_channel, output_channel), 0).item()

                    val_loss += loss * data_idx.shape[0]
                    val_nmse += nmse * data_idx.shape[0]
                
                val_loss /= len(val_feed)
                val_nmse /= len(val_feed)
            all_val_nmse.append(val_nmse)
            print("val_loss={:.6e}".format(val_loss), flush=True)
            print("val_nmse={:.6f}".format(val_nmse), flush=True)
            
        # Write summary
        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("Loss/test", val_loss, epoch)
        writer.add_scalar("NMSE/train", running_nmse, epoch)
        writer.add_scalar("NMSE/test", val_nmse, epoch)
    
    writer.close()

    # Save model
    if config.store_model_path:
        save_model(model, config.store_model_path)   

    return np.mean(all_val_nmse)

def test_process(config):
    device = torch.device(f'cuda:{config.gpu}')

    # Prepare test data
    test_feed = DataFeed_Test(config.test_data_root, config.test_csv)
    test_loader = DataLoader(test_feed, batch_size=1024)

    # Build model
    model, _, _ = build_model(config, device)
    print("Finish building model")

    # Testing
    model.eval()
    with torch.no_grad():
        all_channel_ad_clip_recon = []
        all_channel = []
        all_amplitude = []
        all_phase = []
        all_nmse = []

        for data in test_loader:
            # Get the input
            input_channel = data[0].to(device)
            channel = data[2].to(device)
            channel = torch.unsqueeze(channel, 1)
            amplitude, phase = data[3].to(device), data[4].to(device)

            # Forward
            _, output_channel = model(input_channel)
            nmse = cal_nmse(input_channel, output_channel)
            all_nmse.append(nmse.cpu().numpy())

            output_channel = output_channel.permute(0, 3, 2, 1).contiguous() # [batch, Nt, Nc, RealImag]
            output_channel = torch.view_as_complex(output_channel)
            output_channel = torch.unsqueeze(output_channel, 1)

            amplitude = torch.squeeze(amplitude)
            phase = torch.squeeze(phase)

            all_channel_ad_clip_recon.append(output_channel.cpu().numpy())
            all_channel.append(channel.cpu().numpy())
            all_amplitude.append(amplitude.cpu().numpy())
            all_phase.append(phase.cpu().numpy())

        all_channel_ad_clip_recon = np.concatenate(all_channel_ad_clip_recon)
        all_channel = np.concatenate(all_channel)
        all_amplitude = np.concatenate(all_amplitude)
        all_phase = np.concatenate(all_phase)
        all_nmse = np.concatenate(all_nmse)

    return {
        "all_channel_ad_clip_recon": all_channel_ad_clip_recon,
        "all_channel": all_channel,
        "all_amplitude": all_amplitude,
        "all_phase": all_phase,
        "all_nmse": all_nmse
    }