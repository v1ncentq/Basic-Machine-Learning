import torch, torchvision, os

from torchvision import datasets

from dotenv import load_dotenv

from basic_ml.models import model_efficientnet_b0, model_tinyvgg
from basic_ml.engine import engine


load_dotenv()

device = torch.device('mps')

class_names_path = datasets.ImageFolder(root=(str(os.getenv("IMAGES_TEST_DIR"))))
class_names = class_names_path.classes

model = model_efficientnet_b0.Efficient_B0(output_shape=3)
model_path = str(os.getenv("INF_MODEL_PATH"))
model.load_state_dict(torch.load(f=model_path, map_location=device))

image_path = str(os.getenv("INF_IMAGE_PATH"))

engine.predplot_image(model=model,
                      image_path=image_path,
                      class_names=class_names,
                      image_size=(224,224),
                      device=device)
