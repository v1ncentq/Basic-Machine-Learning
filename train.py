import os
import torch
from dotenv import load_dotenv

from basic_ml.data import dataset
from basic_ml.engine import engine
from basic_ml.models import model_tinyvgg
from basic_ml.utils import utils

load_dotenv()

train_dataset = os.getenv("IMAGES_TRAIN_DIR")
test_dataset = os.getenv("IMAGES_TEST_DIR")

HYPERPARAMETERS = {
    "NUM_EPOCHS": 5,
    "BATCH_SIZE": 32,
    "HIDDEN_UNITS": 10,
    "LEARNING_RAGE" : 0.001
}