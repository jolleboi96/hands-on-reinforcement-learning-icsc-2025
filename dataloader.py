from __future__ import print_function, division
from dataclasses import dataclass
from logging import root
import os
from shutil import move
#from cv2 import INTER_CUBIC
import torch
#import pandas as pd
#from skimage import io #transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#from sac_ae.environments.datamatrix_lookup_class_quad import Datamatrix_lookup_class_quad
import csv
#import cv2
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'



#img_names = data.filepaths

class ToTensor(object):
     """Convert ndarrays in sample to Tensors."""
     def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).unsqueeze(dim=0),
        'labels': torch.from_numpy(labels)}
class Normalize(object):
     """Normalize between [0,1]."""
     def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = image/np.max(image)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #image = image.transpose((2, 0, 1))
        return {'image': image,
        'labels': labels}

class ResizeTri(object):
     """Resize object to shape."""
     def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = cv2.resize(image, dsize =(400,150), interpolation=INTER_CUBIC)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #image = image.transpose((2, 0, 1))
        return {'image': image,
        'labels': labels}

class ResizeQuad(object):
     """Resize object to shape."""
     def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = cv2.resize(image, dsize =(200,150), interpolation=INTER_CUBIC)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #image = image.transpose((2, 0, 1))
        return {'image': image,
        'labels': labels}

class NormalizeLabels(object):
     """Normalize between [0,1]. Must be done if using end tanh activation."""
     def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        image = image/np.max(image)

        labels = labels / 3000 # Normalize by max phase offset in data, = 30(*100) deg.
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #image = image.transpose((2, 0, 1))
        return {'image': image,
        'labels': labels}

class AddNoise(object):
     """
     Add gaussian noise to the image for data augmentation (similar to real acquisitions).
     Mean set to zero, and std gathered approximated from previous live acquisitions. 
     Should be applied after normalization and ToTensor.
     """
     def __call__(self, sample):
        mean = 0
        std = 0.00478
        image, labels = sample['image'], sample['labels']
        if len(image.size()) == 3: # When adding noise to full datamatrix
            _, rows, cols = image.shape
        if len(image.size()) == 2: # When adding noise to profiles
            rows, cols = image.shape
        noise = np.random.default_rng().normal(mean,std,(rows,cols))
        image = image+noise
        return {'image': image,
        'labels': labels}

class TrimEdges(object):
    """
    Trim part of input columns.
    """
    def __call__(self,sample):
        image, labels = sample['image'], sample['labels']
        image = image[:,40:360]
        return {'image': image,
        'labels': labels}

class MoveInjectionCenter(object):
     """
     Sometimes the injected bunch before splitting is not perfectly centered
     in live acquisition. Mimick this by jittering around the simulated bunch a few ns
     to the lift or right during training.
     """
     def __call__(self, sample):
        rng = np.random.default_rng()
        #resolution is 0.5 ns for quad and 1 ns for tri per bin, shift potentially +- 3ns

        shift = rng.integers(-6,6, endpoint=True) # -6,6 for quad, -3,3 for tri
        image, labels = sample['image'], sample['labels']
        #print(shift)
        if shift<0:
            # Add columns to the left and remove from the right
            image = np.pad(image, [(0,0), (abs(shift),0)])
            image = image[:,:-abs(shift)] # Designed for quad
        if shift>0:
            # Add columns to the right and remove from the left
            image = np.pad(image, [(0,0), (0, abs(shift))])
            image = image[:,abs(shift):] # Designed for quad
        return {'image': image,
        'labels': labels}

def random_pad(vec, pad_width, *_, **__):
    mean = 0
    std = 0.00478
    vec[:pad_width[0]] = np.random.default_rng().normal(mean,std)
    vec[vec.size-pad_width[1]:] = np.random.default_rng().normal(mean,std)

class MoveInjectionCenterLiveData(object):
     """
     Used for augmenting live data by moving around injection center. Needs specific
     function as we don't add noise in the same way here.

     Sometimes the injected bunch before splitting is not perfectly centered
     in live acquisition. Mimick this by jittering around the simulated bunch a few ns
     to the lift or right during training.
     """
     def __call__(self, sample):
        rng = np.random.default_rng()
        #resolution is 0.5 ns for quad and 1 ns for tri per bin, shift potentially +- 3ns
       
        shift = rng.integers(-4,4, endpoint=True) # -6,6 for quad, -3,3 for tri
        image, labels = sample['image'], sample['labels']
        #print(shift)
        if shift<0:
            # Add columns to the left and remove from the right. Pad by copying the closest columns.
            image = np.pad(image, [(0,0), (abs(shift),0)], mode=random_pad)
            image = image[:,:-abs(shift)] # Designed for quad
        if shift>0:
            # Add columns to the right and remove from the left
            image = np.pad(image, [(0,0), (0, abs(shift))], mode=random_pad)
            image = image[:,abs(shift):] # Designed for quad
        return {'image': image,
        'labels': labels}



class QuadsplitDataset(Dataset):

    def __init__(self, 
    resize = None, 
    csv_file=None,
    root_dir = None, 
    vpc=False, 
    noise=False,
    norm_labels=False,
    live_data=False,
    move_injection = False,
    use_last_profile = False,
    tri_fixed_voltage = False,
    trim_edges = False,
    transform=transforms.Compose([
        Normalize(),
        ToTensor(),   
    ]),
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.use_last_profile = use_last_profile
        self.tri_fixed_voltage = tri_fixed_voltage

        transform = [Normalize(), ToTensor()] # Default
        
        if resize == 'tri':
            transform.insert(0,ResizeTri()) # Do first
            
        elif resize == 'quad':
            transform.insert(0, ResizeQuad()) # Do first

        if move_injection:
            transform.insert(1,MoveInjectionCenter()) # Do after normalize, before ToTensor()

        if trim_edges:
            transform.insert(1, TrimEdges())

        if norm_labels:
            transform.insert(0,NormalizeLabels())
        
        if noise:
            transform.append(AddNoise()) # Do last

        if live_data:
            transform = [
                    ToTensor()
                ]
            if move_injection:
                transform.insert(0,MoveInjectionCenter()) # Do after normalize, before ToTensor()

            if trim_edges:
                transform.insert(1, TrimEdges())

        
        transform = transforms.Compose(
                transform 
            )
        self.vpc = vpc

        # Load image names of dataset
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            img_names = list(reader) # [label1, label2, data filepath]

        # inputs = []
        # existing_labels = []
        # for entry in img_names:
        #     inputs.append(entry[2])
        #     existing_labels.append([entry[0], entry[1]])
        self.datapaths_and_labels = img_names
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.datapaths_and_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.vpc:
            img_name = self.datapaths_and_labels[idx][-1]
            split = img_name.split('\\')
            img_name = os.path.join('/eos/home-j/jwulff/', split[5], split[6], split[7], split[8], split[9], split[10])
        else:
            img_name = self.datapaths_and_labels[idx][-1] # Always put image last
        try:
            image = np.load(img_name)
        except Exception as e:
            # Sometimes the dataloader fails to load a file stating that the file does not exist (even though it does).
            # Seems connected to losing connection to EOS/internet, was not a problem before...
            print(f'Failed to load {img_name}, waiting for 1 min and retrying...')
            time.sleep(60)
            return self.__getitem__(idx) # Try to re-collect item, hopefully eos has reconnected.
        if self.use_last_profile:
            image = image[-1,:] # Return last profile only instead of full dm.
        
        #image = mpimg.imread(image)
        #image = mpimg.imread(img_name)
        if self.tri_fixed_voltage:
            labels = self.datapaths_and_labels[idx][:-2]
        else:
            labels = self.datapaths_and_labels[idx][:-1]
        labels = [np.float64(i) for i in labels]
        labels = np.asarray(labels)
        #labels = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":

    dataset = QuadsplitDataset(live_data=False, trim_edges=True, move_injection=False, csv_file =r"C:\Users\jwulff\cernbox\workspaces\RL-007-RFinPS\optimising-rf-manipulations-in-the-ps-through-rl\dataset\tri\ref_voltage\dataset_tri_59521_ref.csv")

    fig = plt.figure()

    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['image'].shape, len(sample['labels']))
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('p42, p84: {}, {}'.format(sample['labels'].numpy()[0], sample['labels'].numpy()[1]))
        ax.axis('off')
        plt.imshow(sample['image'][0,:,:])

        #show_landmarks(**sample)

        if i == 3:
            plt.show()
            break
    
    def show_labels_batch(sample_batched):
        """Show images for a batch of samples."""
        images_batch, labels = \
                sample_batched['image'], sample_batched['labels']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
        grid_border_size = 2

        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

        for i in range(batch_size):

            plt.title('Batch from dataloader')
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['labels'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_labels_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

    
        

