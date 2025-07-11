import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class GetITS(Dataset):
    def __init__(self, root_dir, sampling_rate="_5msps", train=True, transform=None,ratio=0.7,snr_db=0.0,normlize=False):
        """
        Args:
            root_dir (string): Directory containing dataset folders.
            sampling_rate (string): Filter files by this sampling rate (e.g., '1msps', '5msps').
            train (bool): If True, load training data (90% of M). If False, load test data (10% of M).
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.root_dir = root_dir
        self.sampling_rate = sampling_rate
        self.train = train
        self.transform = transform
        self.data_samples = None  # Store full dataset concatenated along M
        self.labels = None
        self.classes = sorted(os.listdir(root_dir))  # Class names from folder names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.snr_db = snr_db  # List of SNR values to use
        self.normlize = normlize

        train_data, train_labels = [], []  # Lists to store training data
        test_data, test_labels = [], []  # Lists to store testing data
        # Load all files that match the given sampling rate
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    if file_name.endswith(".mat") and sampling_rate in file_name:
                        file_path = os.path.join(class_path, file_name)
                        label = self.class_to_idx[class_name]

                        # Load .mat file
                        with h5py.File(file_path, 'r') as mat_file:
                            key = list(mat_file.keys())[0]  # Assume first key contains data
                            data = np.array(mat_file[key], dtype=np.float32)  # Convert to NumPy array

                        # Ensure data is at least 2D (N, M)
                        if data.ndim < 2:
                            raise ValueError(f"Invalid data shape in {file_name}: Expected at least 2D, got {data.shape}")

                        N, M = data.shape  # Get original shape

                        # Ensure N is even for (N/2, 2, M) reshaping
                        if N % 2 != 0:
                            raise ValueError(f"Invalid N={N} in {file_name}: N must be even to reshape into (N/2, 2, M)")

                        # Reshape (N, M) → (N/2, 2, M)
                        data = data.reshape(N // 2, 2, M)

                        # **Split each file's M-dimension before concatenation**
                        train_size = int(ratio * M)  # Get `ratio` percentage for training
                        # test_size = int(0.2 * M)  # Get `ratio` percentage for training

                        train_data.append(data[:, :, :train_size])  # First `train_size` for training
                        test_data.append(data[:, :, train_size:])  # Remaining for testing
                        # test_data.append(data[:, :, -test_size:])  # Remaining for testing

                        train_labels.append(np.full((train_size,), label))  # Assign labels
                        test_labels.append(np.full((M - train_size,), label))  # Assign test labels
                        # test_labels.append(np.full((test_size,), label))  # Assign test labels

        if not train_data:
            raise ValueError(f"No valid samples found for sampling rate '{sampling_rate}' in {root_dir}")
        self.train = train
        train_data = np.concatenate(train_data, axis=2)  # Merge all train samples
        test_data = np.concatenate(test_data, axis=2)  # Merge all train samples
        # power = np.mean(np.sum(train_data**2, axis=1))  # real² + imag²
        # scale = np.sqrt(power)
        # **Concatenate all files' train/test data separately**
        if train:
            # power = np.mean(np.sum(train_data**2, axis=1, keepdims=True), axis=0, keepdims=True)
            self.data_samples = train_data  # / np.sqrt(power)
            # self.data_samples = train_data/scale  if self.normlize else train_data # Merge all train samples 
            # self.data_samples = train_data/scale  if self.normlize else train_data # Merge all train samples 
            self.labels = np.concatenate(train_labels, axis=0)  # Merge train labels
        else:
            self.data_samples = test_data #/ np.sqrt(power)
            # self.data_samples = test_data/scale  if self.normlize else test_data # Merge all test samples
            self.labels = np.concatenate(test_labels, axis=0)  # Merge test labels
    def add_noise(self, data):
        """
        Adds white Gaussian noise to the real and imaginary parts separately.
        Args:
            data (numpy array): Signal data (N/2, 2, M) where:
                - data[:, 0, :] is the real part
                - data[:, 1, :] is the imaginary part
        Returns:
            numpy array: Noisy signal with the same shape.
        """
        if self.snr_db is None:
            return data  # No noise added

        data_complex = data[:, 0] + 1j * data[:, 1]
        power_signal = torch.mean(torch.abs(data_complex) ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        power_noise = power_signal / snr_linear
        # Generate AWGN noise
        noise_real = torch.normal(0, torch.sqrt(power_noise / 2), size=data[:, 0].shape)
        noise_imag = torch.normal(0, torch.sqrt(power_noise / 2), size=data[:, 1].shape)
        
        # Create the noisy signal
        data[:, 0] = data[:, 0] + noise_real
        data[:, 1] = data[:, 1] + noise_imag
        return data
    
    def __getitem__(self, index):
        """
        Returns a single sample at the given index.
        """
        sample_data = torch.tensor(self.data_samples[:, :, index], dtype=torch.float32)
        sample_label = torch.tensor(self.labels[index], dtype=torch.long)
        if self.transform:
            sample_data = self.transform(sample_data)
        if not self.train:
            sample_data = self.add_noise(sample_data)
        return sample_data, sample_label

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return self.data_samples.shape[2]
