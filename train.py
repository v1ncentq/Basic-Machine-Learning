import os

import torch
from torchvision import transforms

from dotenv import load_dotenv

from basic_ml.data import dataset
from basic_ml.engine import engine
from basic_ml.models import model_tinyvgg, model_efficientnet_b0
from basic_ml.utils import utils

HYPERPARAMETERS = {
    "NUM_EPOCHS": 10,
    "BATCH_SIZE": 32,
    "HIDDEN_UNITS": 10,
    "LEARNING_RATE" : 0.001,}

TRANSFORMS = {
    "RESIZE64": (64,64),
    "RESIZE224": (224,224)}

def main():
    load_dotenv()

    train_dataset = str(os.getenv("IMAGES_TRAIN_DIR"))
    test_dataset = str(os.getenv("IMAGES_TEST_DIR"))

    metrics = str(os.getenv("METRICS_DIR"))

    device = torch.device('mps')

    data_transform = transforms.Compose([
        transforms.Resize((TRANSFORMS["RESIZE224"])),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])


    train_dataloader, test_dataloader, class_names = dataset.create_dataloaders(
        train_dir=train_dataset,
        test_dir=test_dataset,
        transforms=data_transform,
        batch_size=HYPERPARAMETERS["BATCH_SIZE"])

    model = model_efficientnet_b0.Efficient_B0(
        output_shape=len(class_names),
        freeze_base=True).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=HYPERPARAMETERS["LEARNING_RATE"])

    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=HYPERPARAMETERS["NUM_EPOCHS"],
                device=device,
                metrics=metrics+ "efficientnet_b0.json")

    utils.save_model(model=model,
                    target_dir=metrics,
                    model_name='efficientnet_b0.pth')

if __name__ == "__main__":
    main()
    