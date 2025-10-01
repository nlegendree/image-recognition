from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

def get_dataloaders(data_dir, batch_size=32, train_ratio=0.8, seed=42):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    image_paths = [sample[0] for sample in full_dataset.samples]
    labels = [sample[1] for sample in full_dataset.samples]

    indices = np.arange(len(image_paths))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed
    )

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_paths = [image_paths[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_loader, test_loader, train_paths, test_paths, train_labels, test_labels, full_dataset.classes