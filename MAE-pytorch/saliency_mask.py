import csv
import shelve
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from timm.models.layers import to_2tuple

def retrieve_saliency_map(filename):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    assert(type(filename)==str)
    saliency_dict = {}
    with shelve.open(filename,"r") as att_map_dict:
        print(att_map_dict)
        reader = csv.reader(att_map_dict)
        for row in reader:
            saliency_dict[row[0]] = preprocess(att_map_dict[row[0]])
    return saliency_dict

    '''
    with open(filename,'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            k,v = row[0]
            saliency_dict[str(k)] = float(v)
    return saliency_dict
    '''

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

def convert_saliency_to_mask(saliency_arr,threshold=0.80):
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
    num_keep = math.floor(threshold*len(saliency_arr))
    sorted_indices = np.argsort(saliency_arr)           # sorted saliency indices
    original_indices = np.argsort(sorted_indices)       # indices to recover initial order
    saliency_sorted = saliency_arr[sorted_indices]      # sort the saliency map
    mask = np.zeros(len(saliency_sorted))
    for i in range(num_keep,len(saliency_arr)):
        mask[i] = 1                                     # 1 = mask, 0 = visible
    mask = mask[original_indices]                       # reconvert original order

    
    return mask

class SaliencyMaskGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.input_size = input_size
        self.attn_map_dict = retrieve_saliency_map('/PHShome/pep16/saliency-mae/src/att_map_dict.db')
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self, filename):
        sal = attn_map_dict[filename]
        mask = convert_saliency_to_mask(sal)
        return mask # [196]


if __name__=='__main__':
    attn_map_dict = retrieve_saliency_map('/PHShome/pep16/saliency-mae/src/att_map_dict.db')
    mask = convert_saliency_to_mask(attn_map_dict['n03000684_2453.JPEG'].unsqueeze(0).repeat(8,3,1,1))
    print(mask)
    '''
    # export_csv('att_map_dict_imagenette.csv')
    saliency_arr = np.arange(100)[::-1]
    print(saliency_arr)
    mask = convert_saliency_to_mask(saliency_arr)
    print(mask)
    '''
