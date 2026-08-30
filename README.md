# Basic Machine Learning

A learning project for image classification with **PyTorch**, covering model training, evaluation, and single-image inference.

Based on https://github.com/mrdbourke/pytorch-deep-learning

- **Models:** a custom TinyVGG CNN and pretrained EfficientNet-B0.
- **Training:** `train.py` uses EfficientNet-B0 by default and saves model weights, loss/accuracy metrics, and plots.
- **Inference:** `evaluate.py` displays an image with its predicted class and probability.

**Getting started**

```bash
git clone https://github.com/v1ncentq/Basic-Machine-Learning.git
cd Basic-Machine-Learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Arrange your images in `ImageFolder` format: `data/train/<class>/...` and `data/test/<class>/...`, with matching classes in both folders. Set your paths in a `.env` file at the project root:

```dotenv
IMAGES_TRAIN_DIR=./data/train
IMAGES_TEST_DIR=./data/test
METRICS_DIR=./metrics/
INF_MODEL_PATH=./metrics/efficientnet_b0.pth
INF_IMAGE_PATH=./data/test/class_name/image.jpg
```

Load the environment variables, train the model, then run inference:

```bash
mkdir -p metrics
set -a
source .env
set +a
python train.py
python evaluate.py
```

The current code uses **MPS (macOS)**. To run on CPU or CUDA, update the hardcoded `mps` device settings. `evaluate.py` uses `output_shape=3`; change this to match the number of classes in your trained model.
