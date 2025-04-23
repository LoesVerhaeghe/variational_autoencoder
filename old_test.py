##preprocessing you need to do for canny edge for autoencoder:
def canny_edge(image):
    # Convert the image to grayscale (single channel)
    image_np = np.array(image.convert("L"))  # Convert to grayscale numpy array
    # Apply Canny edge detection
    edges = canny(image_np, sigma=0.8)
    return Image.fromarray(np.uint8(edges * 255))  # Convert back to a PIL Image

# Define the transformation pipeline without Grayscale, because it's handled by canny_edge
transform = transforms.Compose([
    transforms.Resize(IMAGE_DIMENSION),
    transforms.Lambda(canny_edge),  # Apply the custom Canny edge detection
    transforms.ToTensor()  # Convert to tensor
])






####### normalisation
transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_DIMENSION),
    transforms.ToTensor(),  # pixelvalues to range [0,1]
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # Typical normalization values for RGB images
])



from time import time
import multiprocessing as mp
for num_workers in range(2, mp.cpu_count(), 2):  
    train_loader = DataLoader(train_dataset,shuffle=True,num_workers=num_workers,batch_size=32,pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))




    #___________________________________

    #autoencoder with max pooling:
    import torch
import torch.nn as nn

# Defining Autoencoder model
# check literatuur voor gelijkaardige studies die ook niet focussen op orientatie van de features
# check boek over meer fundamentele info over autoencoders
class Autoencoder(nn.Module):
   def __init__(self):
       super().__init__()

       self.encoder = nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), #stride staat default op 1, padding=1 => image verandert niet van grootte
           nn.LeakyReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2), #bij opschuiven geen overlapping omdat je een stride van 2 hebt 
           nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
           nn.LeakyReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2), 
           nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
           nn.LeakyReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2)
        #    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #    nn.LeakyReLU(inplace=True),
        #    nn.MaxPool2d(kernel_size=2, stride=2),
       ) 
       # Latent space
       self.flatten = nn.Flatten()
       self.fc1 = nn.Linear(1*48*64, 1000) 
       self.fc2 = nn.Linear(1000, 1*48*64)
       
       self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2, padding=0, output_padding=0), 
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(1, 128, kernel_size=2, stride=2, padding=0, output_padding=0), 
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2, padding=0, output_padding=0), 
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.Tanh()
        )
       
   def forward(self, x):
        x = self.encoder(x)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = x.view(-1, 1, 48, 64)  # Reshape to match the decoder's input size
        x = self.decoder(x)
        return x
   




   
def visualize_latent_space(model, data_loader, device):
    model.eval()
    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            batch_data = batch.to(device)
            latent_space = model.encoder(batch_data)  # Get latent representation
            reconstructed_data = model(batch_data)  # Autoencoder output (reconstructed images)

            # Move data to CPU for visualization
            original_image = batch_data[0].cpu()
            reconstructed_image = reconstructed_data[0].cpu()
            latent_space_image = latent_space[0].cpu().numpy()

            # Plot original, latent, and reconstructed images
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Plot original image
            axes[0].imshow(original_image.permute(1, 2, 0).numpy())
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Plot latent space (if 2D)
            axes[1].imshow(latent_space_image.squeeze(), cmap='gray')
            axes[1].set_title('Latent Space')
            axes[1].axis('off')

            # Plot reconstructed image
            axes[2].imshow(reconstructed_image.permute(1, 2, 0).numpy())
            axes[2].set_title('Reconstructed Image')
            axes[2].axis('off')

            plt.show()
            break  # Visualize only the first batch
            break  # Visualize only the first batch

# Call the function after training
visualize_latent_space(model, train_loader, device)





with torch.no_grad():
    for batch in train_loader:
        batch_data = batch.to(device)
        reconstructed = model(batch_data)
        batch_data = batch_data.cpu()
        reconstructed = reconstructed.cpu()

        original_im=batch_data[0].permute(1, 2, 0).numpy()
        reconstructed_im=reconstructed[0].permute(1, 2, 0).numpy()
        break

print('original im', original_im)
print('reconstructed im', reconstructed_im)
print(original_im.mean(), original_im.min(), original_im.max())
print(reconstructed_im.mean(), reconstructed_im.min(), reconstructed_im.max())

import logging
logging.debug('This is a debug message')
logging.info('this was done succesfully')
logging.warning('this is taking way to long!')
logging.error('this path was not found')
logging.critical('we cannot establish a connection...')



from utils.helpers import extract_image_paths

paths=extract_image_paths("microscope_images")




class CannyEdgeTransform:
    def __init__(self, sigma=0.8):
        self.sigma = sigma

    def __call__(self, image):
        # Convert image to grayscale numpy array
        image_np = np.array(image.convert("L"))
        # Apply Canny edge detection
        edges = canny(image_np, sigma=self.sigma)
        # Convert back to a PIL Image
        return Image.fromarray(np.uint8(edges * 255))

# Define the transformation pipeline using the custom class
transform = transforms.Compose([
    transforms.Resize(IMAGE_DIMENSION),
    CannyEdgeTransform(sigma=0.8),
    transforms.ToTensor()
])
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_DIMENSION),
    #transforms.ToTensor()
    ])