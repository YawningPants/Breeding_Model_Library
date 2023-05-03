import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image

# Weak augmentation
transform_weak = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.To(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Strong augmentation
transform_strong = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
    transforms.RandomResizedCrop(size=224, scale=(0.6, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mix up
def mixup_data(x, y, alpha=1.0):
    lam = torch.tensor(np.random.beta(alpha, alpha, size=x.size(0)), dtype=torch.float32)
    index = torch.randperm(x.size(0))
    mixed_x = lam.view(-1, 1, 1, 1) * x + (1 - lam.view(-1, 1, 1, 1)) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()

# Example usage
img = Image.open('example.jpg')
img_weak = transform_weak(img)
img_strong = transform_strong(img)
x, y = img_weak.unsqueeze(0), torch.tensor([0])
mixed_x, y_a, y_b, lam = mixup_data(x, y)
