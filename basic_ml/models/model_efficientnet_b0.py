import torch
import torchvision

from torch import nn

class Efficient_B0(nn.Module):
    def __init__(self, output_shape:int, freeze_base: bool=True) -> None:
        super().__init__()

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = torchvision.models.efficientnet_b0(weights=weights)

        if freeze_base:
            for param in self.model.features.parameters():
                param.requires_grad = False

        in_features = self.model.classifier[1].in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=in_features,
                            out_features=output_shape,
                            bias=True)).to(torch.device('mps'))


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)
        