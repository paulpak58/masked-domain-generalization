from collections import Counter
from copy import copy
import csv
import json
import os
import shelve
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from timm.models.layers import to_2tuple
from PIL import Image

def retrieve_saliency_map(filename, num_patches=(16, 16)):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(num_patches),
        transforms.ToTensor(),
    ])
    assert(type(filename)==str)
    img = Image.open(filename)
    img = preprocess(img)
    return np.array(img.squeeze())

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def convert_saliency_to_mask(saliency_arr, num_mask=10):
    '''
    saliency_map: unrolled attention converted to saliency map
    threshold: the percentage of the saliency map to mask
    '''
    
    '''
    with torch.no_grad():
        patch_embed = PatchEmbed()
        patched_map = patch_embed(saliency_map)
        print(patched_map.shape)
    '''
    og_size = copy(saliency_arr.shape)
    saliency_arr = saliency_arr.reshape(-1)
    sorted_indices = np.argsort(saliency_arr)           # sorted saliency indices
    original_indices = np.argsort(sorted_indices)       # indices to recover initial order
    saliency_sorted = saliency_arr[sorted_indices]      # sort the saliency map
    mask = np.zeros(len(saliency_sorted))
    for i in range(num_mask,len(saliency_arr)):
        mask[i] = 1                                     # 1 = mask, 0 = visible
    mask = mask[original_indices]                       # reconvert original order

    mask = mask.reshape(og_size)
    return mask

class SaliencyMaskGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self, filename):
        attn_map_dict = retrieve_saliency_map(filename, num_patches=(self.height, self.width))
        mask = convert_saliency_to_mask(attn_map_dict, self.num_mask)
        return mask # [196]


if __name__=='__main__':
    # attn_map_dict = retrieve_saliency_map('/home/data/att_maps/8e10d0e4-21bc-11ea-a13a-137349068a90.jpg')
    # print(attn_map_dict.shape)
    # mask = convert_saliency_to_mask(attn_map_dict, num_mask=int(196 * 0.8))
    # print(mask.shape)
    # Image.open('/home/data/att_maps/8e10d0e4-21bc-11ea-a13a-137349068a90.jpg').save('bruh2.png')
    # Image.fromarray(mask * 255).convert('1').save('bruh.png')
    # print(os.listdir('/home/data/att_maps')[0])
    thing = json.load(open('/home/data/metadata/iwildcam2021_train_annotations.json'))
    print(thing.keys())
    print(thing['annotations'][0])
