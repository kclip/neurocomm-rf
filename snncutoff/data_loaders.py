from .datasets import *
import warnings 
import torch
from torch.utils.data import  Subset

warnings.filterwarnings('ignore')

def isDVSData(name):
    if 'cifar10-dvs' in name.lower() or 'ncaltech101' in name.lower() or 'dvs128-gesture' in name.lower():
        return True
    return False

def get_data_loaders(path, data,transform=True, resize=False, split_ratio=0.9,args=None):
    if data.lower() == 'cifar10':
        return GetCifar10(path)
    elif data.lower() == 'cifar100':
        return GetCifar100(path)
    elif 'imagenet' in data.lower():
        if 'tiny-imagenet' == data.lower():
            return GetTinyImageNet(path)
        else:
            return GetImageNet(path)
    elif isDVSData(data):
        train_path = path + '/train'
        val_path = path + '/test'
        train_dataset = GetDVS(root=train_path, transform=transform, resize=resize)
        val_dataset = GetDVS(root=val_path, resize=resize)
        return train_dataset, val_dataset

    elif data.lower() == 'shd' or data.lower() == 'ssc':
        train_path = path + '/train'
        val_path = path + '/test'
        train_dataset = GetAudio(root=train_path, transform=transform, resize=resize)
        val_dataset = GetAudio(root=val_path, resize=resize)
        return train_dataset, val_dataset
    elif 'its_' in data.lower():
        sampling_rate = data.lower().split("_")[1]  # Split at "_" and take the second part

        sampling_rate = '_'+sampling_rate
        train_dataset = GetITS(path,sampling_rate=sampling_rate,train=True)
        test_dataset = GetITS(path,sampling_rate=sampling_rate,train=False)
        return train_dataset, test_dataset

    elif data.lower() == 'radioml2018':
        full_dataset = RADIOML2018(path,stft_trans=True,T=args.neuron['T'])
        # Get dataset size and indices
        # dataset_size = len(full_dataset)
        # indices = list(range(dataset_size))
        # random.shuffle(indices)
        # # Create train-test split (80% train, 20% test)
        # split = int(split_ratio * dataset_size)
        # # Split indices into training and test sets (80% train, 20% test)
        # train_indices, test_indices = indices[:split], indices[split:]

        num_classes = 24
        # samples_per_class = 26 * 4096
        samples_per_class = 1 * 4096
        total_samples = num_classes * samples_per_class
        
        # Ensure dataset size matches expectations
        assert len(full_dataset) == total_samples, "Dataset size does not match expected structure."
        
        # Create indices for the entire dataset
        indices = torch.arange(total_samples)
        split_ratio = 0.8
        # Reshape the indices into (num_classes, samples_per_class)
        indices = indices.view(num_classes, samples_per_class)
        
        # Perform a single slicing operation for train and test splits
        split = int(samples_per_class * split_ratio)
        train_indices = indices[:, :split].reshape(-1)  # Take the first part for training
        test_indices = indices[:, split:].reshape(-1)  # Take the remaining part for testing
        
        # Create subsets for training and testing
        train_dataset = Subset(full_dataset, train_indices.tolist())
        test_dataset = Subset(full_dataset, test_indices.tolist())

        # # Create subsets for training and test sets
        # train_dataset = Subset(full_dataset, train_indices)
        # test_dataset = Subset(full_dataset, test_indices)
        return train_dataset, test_dataset
    else:
        NameError("The dataset name is not support!")
        exit(0)
