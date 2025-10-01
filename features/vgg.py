import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

vgg_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

vgg_transform_augmented = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def get_vgg_model(version="vgg13"):
    if version == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
    elif version == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Version must be 'vgg13' or 'vgg19'")

    model = nn.Sequential(*list(model.features.children()))
    model.eval()
    return model


def extract_vgg_feature(img_bgr, model, device="cpu", augment=False):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = vgg_transform_augmented if augment else vgg_transform

    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img_tensor)
        features = torch.flatten(features, 1)

    return features.cpu().numpy().flatten()