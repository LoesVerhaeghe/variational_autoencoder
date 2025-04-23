import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class PerceptualLoss(nn.Module):
    def __init__(self, device, loss_layers_indices=None):
        """
        Initializes the Perceptual Loss module.

        Args:
            device (torch.device): The device (e.g., 'cuda' or 'cpu') to run VGG on.
            loss_layers_indices (list, optional): Indices of VGG19 feature layers to use for loss.
                                                 Defaults to layers before pooling.
            requires_grad (bool): If True, VGG parameters will be updated during training.
                                  Set to False for standard perceptual loss.
        """
        super(PerceptualLoss, self).__init__()
        self.device = device

        # Standard indices for VGG19 layers often used in perceptual loss
        #relu1_2(4), relu2_2(9), relu3_4(18), relu4_4(27), relu5_4(36)
        if loss_layers_indices is None:
            self.loss_layers_indices = [4, 9, 18, 27, 36]
        else:
            self.loss_layers_indices = loss_layers_indices

        # Load pre-trained VGG19 model
        # Use vgg19_bn for batch normalization if preferred
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

        # Extract desired feature layers
        self.feature_layers = nn.ModuleList()
        last_layer_idx = 0
        for idx in self.loss_layers_indices:
            # Create sequential modules up to the desired layer index
            layer_group = nn.Sequential(*list(vgg.children())[last_layer_idx:idx+1])
            self.feature_layers.append(layer_group)
            last_layer_idx = idx + 1 # Move to the next layer start

        # Freeze VGG parameters 
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization transform 
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # Loss function to compare features (L1 or MSE)
        self.criterion = nn.L1Loss().to(device) # L1 is often preferred

    def _preprocess_image(self, img):
        """Prepares image for VGG: handles channels and normalization."""
        # Ensure input is on the correct device
        img = img.to(self.device)

        # If grayscale (1 channel), repeat to make it 3 channels
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1) # Repeat channel dim

        # Apply ImageNet normalization
        img = self.normalize(img)
        return img

    def forward(self, generated_img, target_img):
        """
        Calculates the perceptual loss between generated and target images.

        Args:
            generated_img (torch.Tensor): Output from the autoencoder (B, C, H, W).
            target_img (torch.Tensor): Original input image (B, C, H, W).

        Returns:
            torch.Tensor: The calculated perceptual loss.
        """
        # Preprocess both images
        generated_img_processed = self._preprocess_image(generated_img)
        target_img_processed = self._preprocess_image(target_img)

        # Extract features and calculate loss for each specified layer
        total_loss = 0.0
        current_features_gen = generated_img_processed
        current_features_target = target_img_processed

        for layer in self.feature_layers:
            # Pass through the next block of VGG layers
            current_features_gen = layer(current_features_gen)
            current_features_target = layer(current_features_target)

            # Calculate loss for this layer's features
            layer_loss = self.criterion(current_features_gen, current_features_target)
            total_loss += layer_loss

        return total_loss / len(self.feature_layers) # Average loss over layers