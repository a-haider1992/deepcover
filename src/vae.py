import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import CustomImageDataset
from torch.utils.data import DataLoader
import time
import logging
import matplotlib.pyplot as plt
from PIL import Image
import pdb
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
import optuna

logging.basicConfig(filename='training_vae.log', level=logging.INFO)

class VAE_1(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_1, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = models.resnet50(pretrained=True)

        # Freeze all the layers in the pretrained model
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.encoder.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Linear(512, 2 * latent_dim)  # 2 * latent_dim outputs for mean and variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            # nn.ConvTranspose2d(latent_dim, 512, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3 * 128 * 128),  # Output size matches input size (3, 2100, 2100)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=1)  # Split into mean and log variance
        z = self.reparameterize(mu, logvar)

        # Decoder
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, 3, 128, 128)  # Reshape to match input size
        return x_recon, mu, logvar

# # Example usage:
# latent_dim = 100
# vae = VAE(latent_dim)
# input_data = torch.randn(32, 3, 2100, 2100)  # Example input data
# output, mu, logvar = vae(input_data)
# print("Output shape:", output.shape)

input_dim = (3, 128, 128)  # RGB images
latent_dim = 300
learning_rate = 1e-3
batch_size = 32
epochs = 50

# Initialize TensorBoard writer
writer_res = SummaryWriter()

# Load your custom dataset using ImageFolder
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize([input_dim[1], input_dim[2]]),
    transforms.ToTensor(),
    ])

# Define the loss function
def loss_function(recon_x, x, mu=None, logvar=None):
    # BCE = nn.BCELoss(reduction='sum')
    MSE = nn.MSELoss(reduction='sum')
    recon_x = recon_x.view(-1, 3 * 128 * 128)
    x = x.view(-1, 3 * 128 * 128)
    reconstruction_loss = MSE(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # writer_res.add_scalar('Loss/reconstruction', reconstruction_loss, epoch)
    # writer_res.add_scalar('Loss/kl_divergence', kl_divergence, epoch)
    logging.info(f'Total loss {reconstruction_loss + kl_divergence}')
    return reconstruction_loss + kl_divergence

# pdb.set_trace()
# dataset = CustomImageDataset(txt_file="fundus_train.txt",root_dir="data", transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Define the trainable function
def train_vae(config):
    # Set the batch_size parameter
    batch_size = config["batch_size"]
    latent_dim = config["latent_dim"]
    print(f"Using Batch size: {batch_size}")
    print(f"Using Latent dim: {latent_dim}")
    
    # Update the batch_size in the dataloader
    train_dataset = CustomImageDataset(txt_file="fundus_train.txt",root_dir="data", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CustomImageDataset(txt_file="fundus_test.txt",root_dir="data", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the VAE model
    vae = VAE_1(latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    if torch.cuda.device_count() > 1:
        vae = nn.DataParallel(vae)
    
    # Define the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    total_loss = 0.0
    # Training loop
    vae.train()
    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            data = data[0].to(device)  # Move data to GPU if available
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            if data.shape == recon_batch.shape:
                loss = loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item() / len(data)))
            else:
                print(f'Input batch shape: {data.shape}')
                print(f'Reconstructed batch shape: {recon_batch.shape}')
                raise Exception("Shape of input and reconstructed input does not match!!")
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / (len(train_loader.dataset) * epochs)))
    # average_loss = total_loss / (len(train_loader.dataset) * epochs)
    # Evaluate the trained model
    vae.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data[0].to(device)  # Move data to GPU if available
            recon_batch, mu, logvar = vae(data)
            if data.shape == recon_batch.shape:
                loss = loss_function(recon_batch, data, mu, logvar)
                total_loss += loss.item()
            else:
                print(f'Input batch shape: {data.shape}')
                print(f'Reconstructed batch shape: {recon_batch.shape}')
                raise Exception("Shape of input and reconstructed input does not match!!")
    average_loss = total_loss / len(test_loader.dataset)
    test_accuracy = 1 - average_loss
    print(f"Test Accuracy: {test_accuracy}")
    return average_loss

# Define the search space
# Define the objective function to optimize
def objective(trial):
    # Define the search space
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    # Define additional parameters
    latent_dim = trial.suggest_int("latent_dim", 100, 500)
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    # Train the VAE model with the specified configuration
    config = {"batch_size": batch_size, "latent_dim": latent_dim}
    loss = train_vae(config)
    return loss

# Create an Optuna study
study = optuna.create_study(storage="sqlite:///db.sqlite3", direction="minimize", study_name="vae_study")

# Optimize the objective function
study.optimize(objective, n_trials=5)
print(f"Best value: {study.best_value} (params: {study.best_params})")


# # Initialize the VAE model
# vae = VAE_1(latent_dim)

# # Define the optimizer
# optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
# # Define the stochastic gradient optimizer
# # optimizer = optim.SGD(vae.parameters(), lr=learning_rate, momentum=0.9)
# # Implement model weight decay during training
# weight_decay = 1e-4
# optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
# print(f"Device: {device}")
# vae.to(device)

# # Use DataParallel to utilize multiple GPUs if available
# if torch.cuda.device_count() > 1:
#     vae = nn.DataParallel(vae)

# # Training loop
# vae.train()
# logging.info(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
# for epoch in range(epochs):
#     total_loss = 0.0
#     for batch_idx, data in enumerate(dataloader):
#         # pdb.set_trace()
#         data = data[0].to(device)  # Move data to GPU if available
#         optimizer.zero_grad()
#         recon_batch, mu, logvar = vae(data)
#         if data.shape == recon_batch.shape:
#             loss = loss_function(recon_batch, data, mu, logvar)
#             loss.backward()
#             total_loss += loss.item()
#             optimizer.step()
            
#             if batch_idx % 100 == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(dataloader.dataset),
#                     100. * batch_idx / len(dataloader), loss.item() / len(data)))
#         else:
#             print(f'Input batch shape: {data.shape}')
#             print(f'Reconstructed batch shape: {recon_batch.shape}')
#             raise Exception("Shape of input and reconstructed input does not match!!")
#         # writer_res.add_scalar('Loss/train_batch', loss.item() / len(data), epoch * len(dataloader) + batch_idx)
#     print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / len(dataloader.dataset)))
#     scheduler.step()
#     # print("Decayed learning rate:", scheduler.get_lr())
#     # writer.add_scalar('Loss/train', total_loss / len(dataloader.dataset), epoch)

# # Save the trained model
# timestamp = time.strftime("%Y%m%d%H%M%S")
# model_path = f'vae_model_{timestamp}.pth'
# torch.save(vae.state_dict(), model_path)
# # Close TensorBoard writer
# writer_res.close()

# pdb.set_trace()
# # Evaluate the trained model
# # Load the trained model
# trained_model = VAE_1(latent_dim)
# trained_model.load_state_dict(torch.load('vae_model.pth'))
# trained_model.eval()

# # Load a test image
# test_image_path = "/path/to/test/image.jpg"  # Replace with the path to the user-provided test image

# test_image = Image.open(test_image_path)
# test_image = transform(test_image).unsqueeze(0)  # Apply the same transform as the dataset and add batch dimension

# test_image = test_image.to(device)
# # Reconstruct the test image using the trained model
# with torch.no_grad():

#     reconstructed_image, _, _ = trained_model(test_image)
# # Display the original and reconstructed images

# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(test_image.squeeze().permute(1, 2, 0).cpu().numpy())
# axes[0].set_title('Original Image')
# axes[0].axis('off')
# axes[1].imshow(reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy())
# axes[1].set_title('Reconstructed Image')
# axes[1].axis('off')

# plt.show()
