import csv
import numpy as np
import math

def export_csv(filename):
    assert(type(filename)==str)
    saliency_dict = {}
    with open(filename,'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            k,v = row[0]
            saliency_dict[str(k)] = float(v)
    return saliency_dict

def convert_saliency_to_mask(saliency_arr,threshold=0.80):
    '''
    saliency_arr: unrolled attention converted to saliency map
    threshold: the percentage of the saliency map to mask
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
        if not instance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]


def create_batched_masks(dataloader):


def export_csv_2(filename):
    assert(type(filename)==str)
    saliency_dict = {}
    with open(filename,'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            arr = row[0].split('"')
            fn = arr[0][:-1]        # removing \t character
            for element in arr[1].split(" "):
                if '[' in element:
                    element = element.replace('[','')
                if ']' in element:
                    element = element.replace(']','')
                if element!='' and element!='...':
                    saliency_dict[fn] = float(element)
    return saliency_dict


if __name__=='__main__':
    # export_csv('att_map_dict_imagenette.csv')
    saliency_arr = np.arange(100)[::-1]
    print(saliency_arr)
    mask = convert_saliency_to_mask(saliency_arr)
    print(mask)
