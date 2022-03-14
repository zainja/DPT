import os
import torch
import numpy as np
import cv2
def pck_metric(true_coords, predicted_coords, root, scale, threshold=10):
    batch_size = true_coords.size()[0]
    true_coords = torch.reshape(true_coords, (batch_size, 21, 3))
    predicted_coords = torch.reshape(predicted_coords, (batch_size, 21, 3))
    pck_vals = []
    for i in range(batch_size):
        shifted_predicted_coords = (predicted_coords[i] * scale[i]) + root[i]
        diff = true_coords - shifted_predicted_coords
        distance = torch.sqrt(torch.sum(torch.square(diff), axis=-1))
        pck_vals.append(torch.Tensor.float(distance <= threshold).mean())
    return torch.tensor(pck_vals).mean()

def count_files(directory):
    return len([name for name in os.listdir(directory)])

def process_coords_file(coords_text):
    coords_list = coords_text.split(",")
    coords = []
    for i in range(len(coords_list)):
        coords.append(coords_list[i])
    return np.array(coords).astype("float32")

def get_root_and_scale(absolute_coords):

    M0 = absolute_coords[9]
    absolute_coords = absolute_coords - M0
    W0 = absolute_coords[0]
    distance = np.sqrt(W0.dot(W0))
    return M0, distance

def heatmap(size, uv_coords, sigma, depth_vals=None):
    uv_coords =  uv_coords.astype(np.int32)
    heatmap_layers = np.zeros(size)
    heatmap_layers = np.expand_dims(heatmap_layers, -1)
    heatmap_layers = np.tile(heatmap_layers, (1,1,21))
    heatmap_layers = list(heatmap_layers)

    for i, uv in enumerate(uv_coords):
        heatmap_layers[uv[1]][uv[0]][i] = 1 if depth_vals is None else depth_vals[i] * 100
    heatmap_layers = np.array(heatmap_layers)
    heatmap_layers = cv2.GaussianBlur(heatmap_layers, (0,0), sigma)
    return heatmap_layers