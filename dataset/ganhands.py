import os
from matplotlib.pyplot import sca
import numpy as np
from torch.utils import data
from PIL import Image
from .helpers import count_files, heatmap, process_coords_file, get_root_and_scale

class GanHands(data.Dataset):
    def __init__(self, directory, training, transform, training_split=0.8, depth_maps=False):
        self.directory = directory
        self.depth_maps = depth_maps
        NO_OBJECT_SIZE_TRAIN = int(count_files(os.path.join(directory, "noObject")) * training_split)
        NO_OBJECT_SIZE_VALIDATE = int(count_files(os.path.join(directory, "noObject")) * (1 - training_split))
        OBJECT_SIZE_TRAIN = int(count_files(os.path.join(directory, "withObject")) * training_split)
        OBJECT_SIZE_VALIDATE = int(count_files(os.path.join(directory, "withObject")) * (1 - training_split))

        self.labels = []
        if training:
            for folder in range(1, NO_OBJECT_SIZE_TRAIN):
                folder_path = os.path.join(directory, "noObject", "{:04d}".format(folder))
                objects_count =  count_files(folder_path) // 5
                for i in range(1, objects_count + 1):
                    self.labels.append("noObject/{:04d}/{:04d}".format(folder, i))
            for folder in range(1, OBJECT_SIZE_TRAIN):
                folder_path = os.path.join(directory, "withObject", "{:04d}".format(folder))
                objects_count =  count_files(folder_path) // 5
                for i in range(1, objects_count + 1):
                    self.labels.append("withObject/{:04d}/{:04d}".format(folder, i))
        else:
            for folder in range(NO_OBJECT_SIZE_TRAIN, NO_OBJECT_SIZE_TRAIN + NO_OBJECT_SIZE_VALIDATE):
                folder_path = os.path.join(directory, "noObject", "{:04d}".format(folder))
                objects_count =  count_files(folder_path) // 5
                for i in range(1, objects_count + 1):
                    self.labels.append("noObject/{:04d}/{:04d}".format(folder, i))
            for folder in range(OBJECT_SIZE_TRAIN, OBJECT_SIZE_TRAIN + OBJECT_SIZE_VALIDATE):
                folder_path = os.path.join(directory, "withObject", "{:04d}".format(folder))
                objects_count =  count_files(folder_path) // 5
                for i in range(1, objects_count + 1):
                    self.labels.append("withObject/{:04d}/{:04d}".format(folder, i))
                    
        self.__transform = transform
        self.__length = len(self.labels)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):

        image_file = self.labels[index] + "_color_composed.png"
        image = np.asarray(Image.open(os.path.join(self.directory, image_file)))

        relative_coords_file = self.labels[index] + "_joint_pos.txt"
        absolute_coords_file =self.labels[index] + "_joint_pos_global.txt"
        uv_coords_file = self.labels[index] + "_joint2D.txt"
        
        uv_coords_file = open(os.path.join(self.directory, uv_coords_file)).read()
        uv_coords = process_coords_file(uv_coords_file)
        uv_coords = np.reshape(uv_coords, (21, 2))
        
        relative_coords_file = open(os.path.join(self.directory, relative_coords_file)).read()
        rel_coords = process_coords_file(relative_coords_file)
        rel_coords = np.reshape(rel_coords, (21, 3))

        absolute_coords_file = open(os.path.join(self.directory, absolute_coords_file)).read()
        absolute_coords = process_coords_file(absolute_coords_file)
        absolute_coords = np.reshape(absolute_coords, (21, 3))
        root, scale = get_root_and_scale(absolute_coords)
            
        depth_vals = rel_coords[:, -1] if self.depth_maps else None
        hmap = heatmap(image.shape[:-1], uv_coords, 25, depth_vals)
        sample = {
            "image": image,
            "heatmap": hmap,
            "uv_coords": uv_coords,
            "rel_coords": rel_coords,
            "absolute_coords": absolute_coords,
            "root": root,
            "scale": scale
        }
        # transforms
        if self.__transform is not None:
            sample = self.__transform(sample)

        return sample