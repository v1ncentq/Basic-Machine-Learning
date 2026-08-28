import os

import torch
from torchvision import transforms

from dotenv import load_dotenv

from basic_ml.data import dataset
from basic_ml.engine import engine
from basic_ml.models import model_tinyvgg
from basic_ml.utils import utils

HYPERPARAMETERS = {
    "NUM_EPOCHS": 100,
    "BATCH_SIZE": 64,
    "HIDDEN_UNITS": 100,
    "LEARNING_RATE" : 0.00001}

def main():
    load_dotenv()

    train_dataset = str(os.getenv("IMAGES_TRAIN_DIR"))
    test_dataset = str(os.getenv("IMAGES_TEST_DIR"))

    device = torch.device('mps')

    data_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5)])


    train_dataloader, test_dataloader, class_names = dataset.create_dataloaders(
        train_dir=train_dataset,
        test_dir=test_dataset,
        transforms=data_transform,
        batch_size=HYPERPARAMETERS["BATCH_SIZE"])

    model = model_tinyvgg.TinyVGG(
        input_shape=3,
        hidden_units=HYPERPARAMETERS['HIDDEN_UNITS'],
        output_shape=len(class_names)
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=HYPERPARAMETERS["LEARNING_RATE"])

    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=HYPERPARAMETERS["NUM_EPOCHS"],
                device=device)

    utils.save_model(model=model,
                    target_dir='basic_ml/models/',
                    model_name='tinyvgg.pth')

if __name__ == "__main__":
    main()
    