import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUMWORKERS = 0

def create_dataloaders(
        train_dir:str,
        test_dir:str,
        transforms:transforms.Compose,
        batch_size: int,
        num_workers:int=NUMWORKERS
):
    
    train_data = datasets.ImageFolder(train_dir, transform=transforms)
    test_data = datasets.ImageFolder(test_dir, transform=transforms)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader, class_names
    
