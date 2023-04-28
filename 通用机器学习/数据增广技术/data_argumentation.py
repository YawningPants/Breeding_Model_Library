import torch
import torchvision.transforms as transforms
from PIL import Image

# 加载图像
image = Image.open('example.jpg')

# 定义数据增广器
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 对图像进行增广
augmented_image = transform(image)

# 将增广后的图像转换为PIL图像
augmented_image = transforms.ToPILImage()(augmented_image)

# 显示增广后的图像
augmented_image.show()