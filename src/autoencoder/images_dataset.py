from utils.helpers import extract_image_paths
from torch.utils.data import Dataset
import torch
import cv2

class MicroscopicImages(Dataset):
    def __init__(self, root, magnification, transform=None):
        """
        Args:
            root (str): Path to the root directory containing images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.images = extract_image_paths(root, magnification=magnification)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = torch.load(image_path).to(torch.float32)

        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = torch.tensor(image).permute(2, 0, 1)
        # image = image.float() / 255.0

        if image is None:
            print(f"Failed to load image at index {idx} (path: {image_path})")
            return None

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for image at index {idx}: {e}")
                return None

        return image