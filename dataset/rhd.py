import numpy as np
import torch.utils.data as data
import pickle
import cv2
import os
from PIL import Image

def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2 ** 8 + bottom_bits).astype("float32")
    depth_map /= 2 ** 16 - 1
    depth_map *= 5.0
    return depth_map


class RHD(data.Dataset):
    def __init__(self, img_dir, anno_path, transform):

        self.img_dir = img_dir
        self.anno_dist = pickle.load(open(anno_path, "rb"))
        self.labels = [""] * len(self.anno_dist)
        for i in range(len(self.anno_dist)):
            self.labels[i] = "{:05d}.png".format(i)
        self.__transform = transform
        self.__length = len(self.labels)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        # image
        image = Image.open(os.path.join(self.img_dir, "color", self.labels[index]))
        image = np.array(image)
        image = image / 255

        # depth
        depth = Image.open(os.path.join(self.img_dir, "depth", self.labels[index]))
        depth = np.array(depth)
        depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])

        # sample
        sample = {"image": image, "depth": depth}

        # transforms
        if self.__transform is not None:
            sample = self.__transform(sample)

        return sample
