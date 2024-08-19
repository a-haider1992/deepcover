
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from pytorch_ssim import ssim
import logging

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


logging.basicConfig(filename='vae.log', level=logging.INFO)
logging.info('Started training the OCT_FUNDUS Foundation model')

# Dataset class to handle paired images
class PairedOCTFundusDataset(Dataset):
    def __init__(self, oct_dir, fundus_dir, transform=None):
        self.oct_images = sorted(os.listdir(oct_dir))
        self.fundus_images = sorted(os.listdir(fundus_dir))
        self.oct_dir = oct_dir
        self.fundus_dir = fundus_dir
        self.transform = transform
        
        # Ensure the datasets are of the same length
        assert len(self.oct_images) == len(self.fundus_images), "Mismatch in number of OCT and Fundus images"

    def __len__(self):
        return len(self.oct_images)

    def __getitem__(self, idx):
        # Load OCT and Fundus images
        oct_img_path = os.path.join(self.oct_dir, self.oct_images[idx])
        fundus_img_path = os.path.join(self.fundus_dir, self.fundus_images[idx])
        
        oct_image = Image.open(oct_img_path).convert('RGB')
        fundus_image = Image.open(fundus_img_path).convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            oct_image = self.transform(oct_image)
            fundus_image = self.transform(fundus_image)
        
        return oct_image, fundus_image

# Vision Transformer-based encoder with weight sharing
class SharedViTEncoder(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', embed_dim=768, shared_depth=12):
        super(SharedViTEncoder, self).__init__()
        self.shared_encoder = timm.create_model(vit_model_name, pretrained=True)
        self.shared_blocks = nn.ModuleList(self.shared_encoder.blocks[:shared_depth])
        self.oct_blocks = nn.ModuleList(self.shared_encoder.blocks[shared_depth:])
        self.fundus_blocks = nn.ModuleList(self.shared_encoder.blocks[shared_depth:])
        self.patch_embed = self.shared_encoder.patch_embed
        self.cls_token = self.shared_encoder.cls_token
        self.pos_embed = self.shared_encoder.pos_embed
        self.pos_drop = self.shared_encoder.pos_drop
        self.norm = self.shared_encoder.norm

    def forward(self, oct_img, fundus_img):
        B = oct_img.shape[0]
        oct_x = self.patch_embed(oct_img)
        fundus_x = self.patch_embed(fundus_img)
        oct_x = torch.cat((self.cls_token.expand(B, -1, -1), oct_x), dim=1)
        fundus_x = torch.cat((self.cls_token.expand(B, -1, -1), fundus_x), dim=1)
        oct_x = self.pos_drop(oct_x + self.pos_embed)
        fundus_x = self.pos_drop(fundus_x + self.pos_embed)
        for blk in self.shared_blocks:
            oct_x = blk(oct_x)
            fundus_x = blk(fundus_x)
        for blk in self.oct_blocks:
            oct_x = blk(oct_x)
        for blk in self.fundus_blocks:
            fundus_x = blk(fundus_x)
        oct_features = self.norm(oct_x)
        fundus_features = self.norm(fundus_x)
        return oct_features[:, 0], fundus_features[:, 0]

# Linear Decoder
class LinearDecoder(nn.Module):
    def __init__(self, embed_dim=768, img_size=224, num_channels=3):
        super(LinearDecoder, self).__init__()
        self.fc = nn.Linear(embed_dim, img_size * img_size * num_channels)
        self.img_size = img_size
        self.num_channels = num_channels

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.num_channels, self.img_size, self.img_size)
        return x

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', embed_dim=768, img_size=224, num_channels=3, shared_depth=12):
        super(Autoencoder, self).__init__()
        self.encoder = SharedViTEncoder(vit_model_name=vit_model_name, embed_dim=embed_dim, shared_depth=shared_depth)
        self.oct_decoder = LinearDecoder(embed_dim=embed_dim, img_size=img_size, num_channels=num_channels)
        self.fundus_decoder = LinearDecoder(embed_dim=embed_dim, img_size=img_size, num_channels=num_channels)

    def forward(self, oct_img, fundus_img):
        oct_features, fundus_features = self.encoder(oct_img, fundus_img)
        reconstructed_oct = self.oct_decoder(oct_features)
        reconstructed_fundus = self.fundus_decoder(fundus_features)
        return reconstructed_oct, reconstructed_fundus

# Loss Function
def loss_function(reconstructed_oct, target_oct, reconstructed_fundus, target_fundus):
    mse_loss_oct = F.mse_loss(reconstructed_oct, target_oct)
    mse_loss_fundus = F.mse_loss(reconstructed_fundus, target_fundus)
    ssim_loss_oct = 1 - ssim(reconstructed_oct, target_oct)
    ssim_loss_fundus = 1 - ssim(reconstructed_fundus, target_fundus)
    total_loss = mse_loss_oct + ssim_loss_oct + mse_loss_fundus + ssim_loss_fundus
    return total_loss

# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example transform: resizing, normalization, etc.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define dataset and dataloader
    oct_dir = 'data/OCT'
    fundus_dir = 'data/Fundus'
    dataset = PairedOCTFundusDataset(oct_dir=oct_dir, fundus_dir=fundus_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # Initialize the autoencoder
    autoencoder = Autoencoder(img_size=224, num_channels=3).to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(10):  # Example: 10 epochs
        autoencoder.train()
        for oct_imgs, fundus_imgs in dataloader:
            oct_imgs = oct_imgs.to(device)
            fundus_imgs = fundus_imgs.to(device)

            # Forward pass
            reconstructed_oct, reconstructed_fundus = autoencoder(oct_imgs, fundus_imgs)
            
            # Compute loss
            loss = loss_function(reconstructed_oct, oct_imgs, reconstructed_fundus, fundus_imgs)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")
