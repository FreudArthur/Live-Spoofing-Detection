import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from spoof_model import build_transforms, create_model, save_checkpoint


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def run_epoch(model, dataloader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += accuracy(logits, labels)
        total_batches += 1

    return total_loss / total_batches, total_acc / total_batches


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an ImageNet model for spoof detection")
    parser.add_argument("--data-dir", required=True, help="Dataset root with train/ and val/ folders")
    parser.add_argument("--output", default="checkpoints/best.pt", help="Path to output checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_dataset = ImageFolder(train_dir, transform=build_transforms(train=True, image_size=args.image_size))
    val_dataset = ImageFolder(val_dir, transform=build_transforms(train=False, image_size=args.image_size))
    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_model(num_classes=len(class_names), pretrained=not args.no_pretrained).to(args.device)
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=args.device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=args.device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, class_names, args.output)
            print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
