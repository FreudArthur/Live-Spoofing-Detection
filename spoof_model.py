from pathlib import Path
from typing import Dict, List, Tuple, cast

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CLASS_NAMES = ["non-spoof", "spoof"]


def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_transforms(train: bool = False, image_size: int = 224) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                # Variations de couleur : crucial pour distinguer les attaques
                # écran/imprimé qui dégradent ou décalent les couleurs du visage.
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                # Efface aléatoirement une petite zone : rend le modèle plus robuste
                # et moins dépendant des artefacts locaux.
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def save_checkpoint(model: nn.Module, class_names: List[str], checkpoint_path: str) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }
    output_path = Path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> Tuple[nn.Module, List[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    model = create_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def predict_pil_image(
    model: nn.Module,
    image: Image.Image,
    class_names: List[str],
    device: str = "cpu",
    image_size: int = 224,
) -> Tuple[str, float, Dict[str, float]]:
    transform = build_transforms(train=False, image_size=image_size)
    img_tensor = cast(torch.Tensor, transform(image.convert("RGB")))
    input_tensor = img_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_idx = int(torch.argmax(probs).item())
    top_label = class_names[top_idx]
    top_score = float(probs[top_idx].item())
    class_scores = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}
    return top_label, top_score, class_scores
