import torch
from torch.utils.data import DataLoader
from dataset import CustomImageDataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from piqa import SSIM

import torch.nn as nn
import torch.optim as optim
import timm
import logging

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, swin_encoder):
        super(Autoencoder, self).__init__()
        self.encoder = swin_encoder
        self.encoder.head.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 350)  # Adjust this as needed for your latent dimension
        )
        self.decoder = nn.Sequential(
            nn.Linear(350, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3 * 224 * 224),  # Output size matches input size (3, 224, 224)
        )

    def forward(self, x):
        # Forward pass through encoder
        encoded = self.encoder(x)
        
        # Forward pass through decoder
        decoded = self.decoder(encoded)
        
        # Reshape the decoded output to (batch_size, 3, 224, 224)
        decoded = decoded.view(-1, 3, 224, 224)
        
        return decoded

class CLAHETransform:
    def __init__(self, clip_limit=5.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        # Convert Tensor to PIL image if it's a tensor
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)

        # Convert PIL image to numpy array
        img = np.array(img)
        if len(img.shape) == 2:  # If the image is grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply CLAHE to each channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Convert numpy array back to PIL image
        img = Image.fromarray(img)
        return img
    
class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


logging.basicConfig(filename='vae.log', level=logging.INFO)
logging.info('Started training the OCT_FUNDUS Foundation model')
# Load the pretrained Swin Transformer encoder
swin_encoder = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)

# Create the autoencoder model
autoencoder = Autoencoder(swin_encoder)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
autoencoder = autoencoder.to(device)

# Wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    autoencoder = nn.DataParallel(autoencoder)

# Define the loss function
mse_loss = nn.MSELoss()

# Define the optimizer
optimizer = optim.SGD(autoencoder.parameters(), lr=1e-2, momentum=0.9)

image_size = 256  # Resize to 256x256
crop_size = 224  # Center crop to 224x224

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize([image_size, image_size]),
    transforms.CenterCrop(crop_size),  # Apply center crop
    CLAHETransform(clip_limit=1.0, tile_grid_size=(8, 8)),  # Apply CLAHE
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the datasets and data loaders for fundus and OCT images
fundus_dataset = CustomImageDataset(txt_file="fundus_patches.txt", root_dir=".", transform=transform)
oct_dataset = CustomImageDataset(txt_file="annotations-oct-sorted.txt", root_dir=".", transform=transform)

fundus_dataloader = DataLoader(fundus_dataset, batch_size=32)
oct_dataloader = DataLoader(oct_dataset, batch_size=32)

# Define the compute_loss function
def compute_loss(fundus_encoded, fundus_images, oct_encoded, oct_images, mse_loss):
    # fundus_images = fundus_images.view(-1, 3*224*224)
    # oct_images = oct_images.view(-1, 3*224*224)
    # fundus_encoded = fundus_encoded.view(-1, 3*224*224)
    # oct_encoded = oct_encoded.view(-1, 3*224*224)

    mse_loss_value = mse_loss(fundus_encoded, fundus_images) + mse_loss(oct_encoded, oct_images)
    # ssim = SSIMLoss().to(device)
    # ssim_loss = ssim(fundus_encoded, fundus_images) + ssim(oct_encoded, oct_images)
    return mse_loss_value

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for fundus_images, oct_images in zip(fundus_dataloader, oct_dataloader):
        # Move images to device
        fundus_images = fundus_images.to(device)
        oct_images = oct_images.to(device)

        # Forward pass
        fundus_encoded = autoencoder(fundus_images)
        oct_encoded = autoencoder(oct_images)

        # Compute the loss
        total_loss = compute_loss(fundus_encoded, fundus_images, oct_encoded, oct_images, mse_loss)

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}")

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item()}")
    torch.save(autoencoder.state_dict(), "autoencoder_oct_fundus.pth")

# Save the trained autoencoder model
torch.save(autoencoder.state_dict(), "autoencoder_oct_fundus.pth")
