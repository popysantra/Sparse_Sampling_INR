## Sample commamnd: python train.py --dataset_name=isabel --input_data_file=./isabel.vti --varname=P --run_device=cuda:2 --outpath=./models/ --outdata_path=./outputs/ 
# --samp_percentage_to_use=0.1 --samp_strategy=random

## Train Model with a Volume Dataset
##########################################################
import numpy as np
import vtk
import os
import time
import argparse
from vtkmodules.util import numpy_support
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
from updated_model import *
from utils import *
from sampling_strategy import *
##########################################################
## Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', action="store", required=True, help="data set name")
parser.add_argument('--input_data_file', action="store", required=True, help="input volume file in vti format")
parser.add_argument('--varname', action="store", required=True, help="variable name to be trained")
parser.add_argument('--run_device', action="store", required=True, help="cuda device id")
parser.add_argument('--outpath', action="store", required=True, help="outpath for trained models")
parser.add_argument('--outdata_path', action="store", required=True, help="outpath for recon vti files")
parser.add_argument('--samp_percentage_to_use', action="store", required=False, default=0.5, type=float, help="percentage of data to be used for training (between 0 and 1)")
parser.add_argument('--samp_strategy', action="store", required=False, default='random', type=str, help="sampling strategy to use for training data (random or grid)")

## Parse arguments
args = parser.parse_args()
dataset_name = getattr(args, 'dataset_name')
input_data_file = getattr(args, 'input_data_file')
varname = getattr(args, 'varname')
run_device = getattr(args, 'run_device')
outpath = getattr(args, 'outpath')
outdata_path = getattr(args, 'outdata_path')
samp_percentage_to_use = getattr(args, 'samp_percentage_to_use')
samp_strategy = getattr(args, 'samp_strategy')

model_name = dataset_name + '_' + varname + '_'
learning_rate = 0.00005
MAX_EPOCH = 200
BATCH_SIZE = 2048
number_layers = 6
neurons_per_layer = 100
neurons_per_BN_layer = 25
lr_schedule_stepsize = 15
lr_gamma = 0.8
weight_decay = 1e-5
num_input_dim = 3
num_output_dim = 1
n_bins = 10
n_strata = 32
edge_weight = 2.0
group_size = 50000  ## increase this value when running on a GPU
##############################################################################

## Load data
data = read_vti_file(input_data_file)
## Prepare data
np_arr_coord, np_arr_vals = data_setup(data, varname)
## convert to torch tensor
torch_coords = torch.from_numpy(np_arr_coord)
torch_vals = torch.from_numpy(np_arr_vals)

sampling_config = SamplingConfig(
        sampling_type=SamplingType(samp_strategy),
        n_samples= int(data.GetNumberOfPoints()*samp_percentage_to_use),
        n_bins= n_bins, n_strata=n_strata, edge_weight= edge_weight,
        random_seed= 42
    )

## Apply subsampling if desired
if samp_percentage_to_use < 1.0:
    torch_coords, torch_vals = create_sparse_data(torch_coords, torch_vals, sampling_config)
    

# Convert to tensors before DataLoader
torch_coords = torch.from_numpy(torch_coords).float()
torch_vals   = torch.from_numpy(torch_vals).float().unsqueeze(1)  # (N, 1)

## create dataloader
train_dataloader = DataLoader(TensorDataset(torch_coords, 
                                            torch_vals), 
                                            batch_size=BATCH_SIZE,
                                            pin_memory=True, 
                                            shuffle=True)
print('Data setup is complete')
############################################################
device = torch.device(run_device) if torch.cuda.is_available() else torch.device("cpu")

## Prepare model
##########################################################
model = MyResidualSineNet(input_dim=3, hidden_dim=120, num_residual_blocks=10, output_dim=1, omega_0=30).to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size = lr_schedule_stepsize, gamma = lr_gamma)
criterion = nn.MSELoss()
############################################################

## Begin training
##########################################################
train_loss_list = list()
t_begin = time.time()
for epoch in range(MAX_EPOCH+1):

  model.train()

  temp_loss_list = list()
  for X_train, y_train in train_dataloader:
    X_train = X_train.type(torch.float32).to(device)
    y_train = y_train.type(torch.float32).to(device)

    optimizer.zero_grad()
    predictions = model(X_train)

    loss = criterion(predictions,y_train)
    loss.backward()
    optimizer.step()

    temp_loss_list.append(loss.detach().cpu().numpy())

  train_loss_list.append(np.average(temp_loss_list))

  print("epoch: ", epoch," train loss: ", train_loss_list[-1], "LR: ", optimizer.param_groups[0]['lr'])

  scheduler.step()

  if epoch == MAX_EPOCH:
  ## save model
    out_model_name = outpath + model_name + str(epoch) +  '_' + samp_strategy + '_' + str(samp_percentage_to_use) + '.pth'
    torch.save({"epoch": epoch + 1,
              "model_state_dict": model.state_dict()},
              out_model_name)
t_end = time.time()
print('Training is completed in', (t_end - t_begin)/60.0, 'mins')

## Reconstruct and compute PSNR and store vti file
##########################################################
start_time = time.time()  # Start timer for prediction part
final_recon_vals = recon_data(model, np_arr_coord, np_arr_vals, group_size, device)
end_time = time.time()  # End timer for prediction part
total_time = end_time - start_time  # Accumulate time
print('Reconstruction is completed in', total_time, 'secs')

##Compute PSNR and RMSE
psnr = compute_PSNR(final_recon_vals, np.squeeze(np_arr_vals))
rmse = compute_rmse(final_recon_vals, np.squeeze(np_arr_vals))
print('PSNR:', psnr, 'RMSE:', rmse)

##Save the recon file as a vti voulme
save_volume(data, varname, final_recon_vals, outdata_path, dataset_name,samp_percentage_to_use, samp_strategy)