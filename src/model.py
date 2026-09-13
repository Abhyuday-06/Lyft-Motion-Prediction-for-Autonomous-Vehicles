import torch
from torch import nn
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

ARCHS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
}


def build_backbone(architecture: str, in_channels: int, num_outputs: int, pretrained: bool = True) -> nn.Module:
    ctor, weights = ARCHS[architecture]
    model = ctor(weights=weights if pretrained else None)

    old_conv = model.conv1
    new_conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if pretrained:
        # Inflate the RGB filters across the raster's extra history channels.
        # Tiling and rescaling by 3/in_channels keeps the response magnitude of
        # the pretrained stem roughly unchanged for an all-ones input.
        with torch.no_grad():
            w = old_conv.weight  # (64, 3, 7, 7)
            reps = (in_channels + 2) // 3
            tiled = w.repeat(1, reps, 1, 1)[:, :in_channels]
            new_conv.weight.copy_(tiled * (3.0 / in_channels))
    model.conv1 = new_conv

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_outputs)
    return model


class LyftMultiModel(nn.Module):
    def __init__(self, cfg: dict, num_modes: int = 3, pretrained: bool = True):
        super().__init__()
        architecture = cfg["model_params"]["model_architecture"]
        history_num_frames = cfg["model_params"]["history_num_frames"]
        in_channels = 3 + (history_num_frames + 1) * 2

        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.backbone = build_backbone(architecture, in_channels, self.num_preds + num_modes, pretrained)

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        bs = out.shape[0]
        pred, confidences = torch.split(out, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


def pytorch_neg_multi_log_likelihood_batch(
    gt: torch.Tensor,
    pred: torch.Tensor,
    confidences: torch.Tensor,
    avails: torch.Tensor,
) -> torch.Tensor:
    """Negative multi-modal log-likelihood loss, as used in the official l5kit baseline."""
    assert len(pred.shape) == 4
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords)
    assert avails.shape == (batch_size, future_len)
    assert confidences.shape == (batch_size, num_modes)

    gt = torch.unsqueeze(gt, 1)
    avails = avails[:, None, :, None]

    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # (batch, modes, time)
    error = torch.log(confidences + 1e-9) - 0.5 * torch.sum(error, dim=-1)  # (batch, modes)
    error = -torch.logsumexp(error, dim=-1)  # (batch,) ; logsumexp is numerically stable
    return torch.mean(error)
