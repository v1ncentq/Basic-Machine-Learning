import torch, json
from pathlib import Path
import matplotlib.pyplot as plt

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    assert model_name.endswith('.pth') or model_name.endswith('.pt')
    model_save_path = target_dir_path / model_name

    print('Model saved in: {model_save_path}')
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def metrics_plot(metrics_path: str):

    jsonfile = Path(metrics_path)
    with open(jsonfile, "r", encoding="utf-8") as f:
        results = json.load(f)

    keys = {"train_loss", "train_acc", "test_loss", "test_acc"}

    epochs = range(1, len(results["train_loss"])+1)

    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, results["train_loss"], label="Train Loss", marker="o", color="royalblue")
    plt.plot(epochs, results["test_loss"], label="Test Loss", marker="o", color="darkorange")
    plt.title("Loss Curves", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    plt.subplot(1,2,2)
    plt.plot(epochs,results["train_acc"], label= "Test Accuracy", marker="o", color="forestgreen")
    plt.plot(epochs,results["test_acc"], label = "Train Accuracy", marker="o", color="crimson")
    plt.title("Accuracy Curves", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()

    save_image = jsonfile.with_suffix(".png")
    plt.savefig(save_image, dpi=300)



