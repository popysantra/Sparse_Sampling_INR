## Util Functions
##########################################################

import numpy as np
import vtk
import os
from vtkmodules.util import numpy_support
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
from sampling_strategy import SamplingType, SamplingConfig, get_sampling_strategy
from vtk.util.numpy_support import vtk_to_numpy
##########################################################


def load_trained_model(mfile, single_model, device):
    checkpoint = torch.load(mfile, map_location=torch.device(device))
    # Adjust state_dict keys if they were saved using DataParallel
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
      # Remove the 'module.' prefix if it exists
      new_key = key.replace("module.", "") if key.startswith("module.") else key
      new_state_dict[new_key] = value

    single_model.load_state_dict(new_state_dict)
    return single_model

def recon_data(model, np_arr_coord, np_arr_vals, group_size, device):
    model.eval()
    torch_coords = torch.from_numpy(np_arr_coord)
    torch_vals = torch.from_numpy(np_arr_vals)
    predicted_vals = torch.zeros_like(torch_vals).squeeze().to(device)
    with torch.no_grad():
        for i in range(0, torch_coords.shape[0], group_size):
            coords = torch_coords[i:i + group_size]
            coords = coords.type(torch.float32).to(device)
            vals = model(coords)
            predicted_vals[i:i+group_size] = vals.squeeze()

    ## create a single list by combining all list elements
    #extracted_vals1 = [item for sublist in predicted_vals1 for item in sublist]
    #extracted_vals1 = np.asarray(extracted_vals1)

    return predicted_vals.cpu().numpy().squeeze()

def save_volume(data, varname, extracted_vals1, outdata_path, dataset_name, samp_strategy, samp_percentage_to_use):
    # Now scale back to original range and then store
    min_data = data.GetPointData().GetArray(varname).GetRange()[0]
    max_data = data.GetPointData().GetArray(varname).GetRange()[1]
    extracted_vals1 = ((extracted_vals1 + 1) / 2.0) * (max_data - min_data) + min_data
    vtk_arr1 = numpy_support.numpy_to_vtk(extracted_vals1.squeeze())
    vtk_arr1.SetName('recon_' + varname)
    ## create an empty vtkImageData
    outdata = createVtkImageData(data.GetOrigin(), data.GetDimensions(), data.GetSpacing())
    outdata.GetPointData().AddArray(vtk_arr1)
    # Write reconstructed data out
    outfname = os.path.join(outdata_path, 'recon_' + dataset_name + 
                            '_' + varname + '_'+ samp_strategy + '_' + str(samp_percentage_to_use) + '.vti')
    write_vti(outdata, outfname)

def data_setup(data, arrname):

    ## Load data
    num_pts = data.GetNumberOfPoints()
    dims = data.GetDimensions()

    data_arr = data.GetPointData().GetArray(arrname)

    np_arr_coord = np.zeros((num_pts,3))
    np_arr_vals = np.zeros((num_pts,1))

    for i in range(num_pts):
      pt = data.GetPoint(i)
      val1 = data_arr.GetTuple1(i)
      np_arr_vals[i,:] = val1
      np_arr_coord[i,:] = pt

    original_coords = np_arr_coord.copy()
    original_vals = np_arr_vals.copy()
    min_data = np.min(np_arr_vals[:,0])
    max_data = np.max(np_arr_vals[:,0])
    np_arr_vals[:,0] = 2.0*((np_arr_vals[:,0]-min_data)/(max_data-min_data)-0.5)

    ### Normalize between 0 to 1
    np_arr_coord[:,0] = np_arr_coord[:,0]/dims[0]
    np_arr_coord[:,1] = np_arr_coord[:,1]/dims[1]
    np_arr_coord[:,2] = np_arr_coord[:,2]/dims[2]

    return np_arr_coord, np_arr_vals, original_coords, original_vals


def data_setup_vtp(data, arrname, dims, scalar_min_ref=None, scalar_max_ref=None):

    # coordinates
    np_arr_coord = vtk_to_numpy(
        data.GetPoints().GetData()
    ).astype(np.float32)

    # scalar values
    data_arr = data.GetPointData().GetArray(arrname) \
               or data.GetPointData().GetArray(0)

    if data_arr is None:
        raise ValueError(f"No array '{arrname}' found")

    np_arr_vals = vtk_to_numpy(data_arr).astype(np.float32).reshape(-1,1)

    # scalar normalization
    min_data = scalar_min_ref if scalar_min_ref is not None else np_arr_vals.min()
    max_data = scalar_max_ref if scalar_max_ref is not None else np_arr_vals.max()

    np_arr_vals[:,0] = 2.0 * (
        (np_arr_vals[:,0] - min_data) /
        (max_data - min_data + 1e-8) - 0.5
    )

    # coordinate normalization (same as VTI)
    np_arr_coord[:,0] /= dims[0]
    np_arr_coord[:,1] /= dims[1]
    np_arr_coord[:,2] /= dims[2]

    return np_arr_coord, np_arr_vals

def coord_setup(data):

    ## Load data
    num_pts = data.GetNumberOfPoints()
    dims = data.GetDimensions()
    np_arr_coord = np.zeros((num_pts,3))
    for i in range(num_pts):
      pt = data.GetPoint(i)
      np_arr_coord[i,:] = pt

    ### Normalize between 0 to 1
    np_arr_coord[:,0] = np_arr_coord[:,0]/dims[0]
    np_arr_coord[:,1] = np_arr_coord[:,1]/dims[1]
    np_arr_coord[:,2] = np_arr_coord[:,2]/dims[2]

    return np_arr_coord

def random_sampling(dims,samp_percentage_to_use,coords, vals):
    total_pts = dims[0]*dims[1]*dims[2]
    num_samples = int(total_pts*samp_percentage_to_use)
    random_indices = torch.randint(0, total_pts, (num_samples,))
    ## now select corresponding points based on random indices
    return coords[random_indices], vals[random_indices]



def create_sparse_data(coords, values, sampling_config):
    """
    Create sparse observations using the specified sampling strategy.
    `coords` and `values` may be GPU tensors; sampling is done on CPU
    (scipy-based), then the result is returned as numpy arrays so the
    caller can decide where to place them.
    """
    if torch.is_tensor(coords):
        coords = coords.cpu().numpy()
    if torch.is_tensor(values):
        values = values.cpu().numpy()

    strategy = get_sampling_strategy(sampling_config)
    sampled_coords, sampled_vals = strategy.sample(coords, values.ravel())
    return sampled_coords, sampled_vals


# Function to read VTI files and extract data
def read_vti_file(file_path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def read_vtp_file(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def write_vti(data, fname):
    writer = vtk.vtkXMLImageDataWriter()
    #fname = os.path.join(directory, 'recon_' + dataset_name + '.vti')
    writer.SetInputData(data)
    writer.SetFileName(fname)
    writer.Write()

## compute SNR
def compute_PSNR(arrgt, arr_recon):
    diff = arrgt - arr_recon
    sqd_max_diff = (np.max(arrgt) - np.min(arrgt))**2
    snr = 10 * np.log10(sqd_max_diff / np.mean(diff**2))
    return snr

## compute RMSE
def compute_rmse(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    return np.sqrt(mse)

## return an empty vtkimagedata
def createVtkImageData(origin, dimensions, spacing):
    localDataset = vtk.vtkImageData()
    localDataset.SetOrigin(origin)
    localDataset.SetDimensions(dimensions)
    localDataset.SetSpacing(spacing)
    return localDataset





