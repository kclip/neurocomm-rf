# Examples

## Training

```python 
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from snncutoff.layers.snn import SRLayer       # Spiking Regularization layer
from snncutoff.neuron import LIF               # Leaky Integrate-and-Fire neuron model
import snncutoff                             

# Define the CNN model architecture
class TinyVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(TinyVGG, self).__init__()
        # Define the convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()                           # Flatten for fully connected layer
        )
        # Define the fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, num_classes),   # Adjust for the output size of the convolution layers
        )
        
    def forward(self, x):
        x = self.features(x)                  # Pass input through feature extractor
        x = self.classifier(x)                # Pass through classifier to get predictions
        return x


model = TinyVGG(num_classes=10) # Buid an ANN model

#####################################################################
###############  Build an SNN model in snncutoff     ################
#####################################################################

# Configure the spiking neuron layer
neuron_params = {
    'name': 'LIF',                            # Neuron type: Leaky Integrate-and-Fire
    'vthr': 1.0,                              # Voltage threshold
    'T': 2,                                   # Time step
    'delta': 0.5,                             # decay factor
    'mem_init': 0.,                           # Initial membrane potential
    'multistep': True,                        # Multistep simulation
    'reset_mode': 'hard',                     # Hard reset mode for spiking
}

snn_layer = SRLayer(neuron=LIF, regularizer=None, neuron_params=neuron_params) # Configure the spiking layer
snn_model = snncutoff.pre_config(model, snn_layer, method='snn')  # convert ANN model with the spiking layer

#####################################################################
############################# End ##################################
#####################################################################

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss().cuda()                          # Use CrossEntropyLoss for classification
optimizer = torch.optim.Adam(snn_model.parameters(), lr=0.001)    # Adam optimizer

# Data transformations and loading MNIST dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),                  # Convert images to grayscale
    transforms.ToTensor(),                                        # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))                          # Normalize images
])

# Load training and test datasets
train_dataset = datasets.MNIST(root='./datasets', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./datasets', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training loop function
def train_model(model, criterion, optimizer, train_loader, num_epochs=5):
    model.train()                                                 # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()         # Move data to GPU
            images = images.unsqueeze(1)
            images = images.repeat(1, snn_layer.T, 1, 1, 1)       # Repeat for time steps
            images = images.transpose(0, 1)                       # Transpose for time dimension
            optimizer.zero_grad()                                 # Zero gradients
            outputs = model(images)                               # Forward pass
            loss = criterion(outputs.mean(0), labels)             # Compute loss
            loss.backward()                                       # Backpropagation
            optimizer.step()                                      # Update weights
            
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Train the model
train_model(snn_model.to('cuda'), criterion, optimizer, train_loader, num_epochs=5)


```


## Multi-GPU Training

```sh
# Revise my_snn_training.yaml to configure models, datasets and neurons
python training.py --config configs/customize/my_snn_training.yaml  
```