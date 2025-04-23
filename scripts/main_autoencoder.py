from utils.plotting_utilities import visualize_reconstruction
from src.autoencoder.model_structure import Autoencoder
from src.autoencoder.training_autoencoder import train_autoencoder
from src.autoencoder.images_dataset import MicroscopicImages
from src.perceptual_loss import PerceptualLoss # Import the class you just defined
import torch.optim as optim
from torchsummary import summary
import torch
import os
import shutil
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data import ConcatDataset
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


#use GPU if available
torch.cuda.set_device(0) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', DEVICE)

# ### Loss functions
# mse_loss_fn = nn.MSELoss()
ssim_metric = SSIM(data_range=1.0, gaussian_kernel=True, kernel_size=7)

# def combined_loss(reconstructed, original, device):
#     ssim_metric.to(device)
#     mse_loss = 10*mse_loss_fn(reconstructed, original)
#     ssim_val = ssim_metric(reconstructed, original)
#     ssim_loss = 1.0 - ssim_val
#     total_loss = mse_loss + ssim_loss
#     return total_loss, mse_loss, ssim_loss

### perceptual loss
RECONSTRUCTION_LOSS_WEIGHT = 0  # Weight for MSE/SSIM
PERCEPTUAL_LOSS_WEIGHT = 1 # Weight for Perceptual Loss (tune this)

# Perceptual Loss
perceptual_criterion = PerceptualLoss(device=DEVICE).to(DEVICE)

# --- Combined Loss Function ---
def combined_loss(outputs, targets, device):
    # Reconstruction Loss
    ssim_metric.to(device)
    recon_loss = RECONSTRUCTION_LOSS_WEIGHT * ssim_metric(outputs, targets)

    # Perceptual Loss
    perc_loss = PERCEPTUAL_LOSS_WEIGHT * perceptual_criterion(outputs, targets)

    # Combine losses
    total_loss = (recon_loss + perc_loss)

    return total_loss, recon_loss, perc_loss # Return individual components for logging


# load model structure
model=Autoencoder().to(DEVICE)
print(summary(model, input_size=(1, 384, 512)))

# Setting training parameters
RANDOM_SEED= 69
LEARNING_RATE = 0.0001 # bigger than 0.0001 ends in local minima
BATCH_SIZE = 16
NUM_EPOCHS = 1000 # we use early stopping anyway
OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)

torch.manual_seed(RANDOM_SEED)

### load data
base_folder = "data/microscope_images_CLAHE"
dataset_10x = MicroscopicImages(root=base_folder, magnification=10, transform=None)
dataset_40x = MicroscopicImages(root=base_folder, magnification=40, transform=None)

# Concatenate the two datasets into one
dataset = ConcatDataset([dataset_10x, dataset_40x])

train_size = int(0.75 * len(dataset)) 
val_size = len(dataset) - train_size 
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

### Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=10, pin_memory=True, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=10, pin_memory=True, shuffle=False)

### Train autoencoder
trained_model, log_dict=train_autoencoder(NUM_EPOCHS, 
                                          model, 
                                          OPTIMIZER, 
                                          DEVICE, 
                                          train_loader, 
                                          val_loader, 
                                          loss_fn=combined_loss, 
                                          skip_epoch_stats=False, 
                                          plot_losses_path='outputs/losses.png', 
                                          save_model_path='outputs/model_perceptualloss.pt')


visualize_reconstruction(model, DEVICE, train_loader, num_images=5, path='outputs/traindataset_reconstruction.png')
visualize_reconstruction(model, DEVICE, val_loader, num_images=5, path='outputs/validationdataset_reconstruction.png')


#to generate encoded images with trained model, first copy the preprocessed data folder in the output folder and rename, then run this code:

# Copy the folder structure from src_folder to dst_folder
src_folder = 'data/microscope_images_grayscaled'
dst_folder = 'outputs/microscope_images_encoded_perceptuallossonly'
shutil.copytree(src_folder, dst_folder)

def encode_images(magnification, dst_folder):
    # Initialize dataset and dataloader for the given magnification
    dataset = MicroscopicImages(root=dst_folder, magnification=magnification, transform=None)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=10, pin_memory=True, shuffle=False)
    
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(DEVICE)
            encoded_image = model(data, get_encoded=True)
            encoded_image = encoded_image.squeeze(0)  # Remove batch dimension

            # Get the original filename and folder path from the dataset
            original_path = dataset.images[i]  # Adjust based on your dataset class
            folder_path, file_name = os.path.split(original_path)
            new_file_name = file_name.replace(".pt", "_encoded.pt")  # Change extension to .pt

            # Save the encoded tensor and delete the original file
            new_file_path = os.path.join(folder_path, new_file_name)
            torch.save(encoded_image.cpu(), new_file_path)

            if os.path.exists(original_path):
                os.remove(original_path)  # Delete the original file
                
# # Encode and save for both magnifications
encode_images(magnification=10, dst_folder=dst_folder)
encode_images(magnification=40, dst_folder=dst_folder)

# ## plot latent space
# import matplotlib.pyplot as plt
# path='outputs/microscope_images_encoded/2024-03-12/basin5/40x/12125430_encoded.pt'
# image=torch.load(path)

# # Set up the figure and subplots
# num_channels = image.shape[0]  # Number of filters
# fig, axes = plt.subplots(num_channels, 1, figsize=(5, num_channels * 3))
# for i in range(num_channels):
#     axes[i].imshow(image[i, :, :].numpy(), cmap='gray')
#     axes[i].set_title(f'Filter {i+1}')
#     axes[i].axis('off')
# plt.tight_layout()
# plt.show()