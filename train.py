import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix

from spoof_model import build_transforms, create_model, save_checkpoint


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


def run_epoch(model, dataloader, criterion, optimizer=None, device="cpu"):
    """Passe d'entraînement (si optimizer fourni) ou de validation.

    Retourne (loss_moyenne, accuracy_moyenne).
    """
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


def phase_hyperparams(epoch, phase1_epochs, phase2_epochs):
    """Découpe les epochs en deux phases :
    - phase 1 : backbone gelé, on n'entraîne que la tête
    - phase 2 : tout le réseau est entraîné

    Retourne (phase_name, freeze_backbone).
    """
    if epoch <= phase1_epochs:
        return "phase1", True
    return "phase2", False


def evaluate_model(model, dataloader, class_names, device="cpu"):
    """Évaluation complète : accuracy, rapport de classification, matrice de confusion."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = float((torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item())
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, report, cm


def main():
    parser = argparse.ArgumentParser(description="Fine-tune an ImageNet model for spoof detection")
    parser.add_argument("--data-dir", required=True, help="Dataset root with train/ and val/ folders")
    parser.add_argument("--output", default="checkpoints/best.pt", help="Path to output checkpoint")
    parser.add_argument("--epochs", type=int, default=12, help="Total epochs (phase 1 + phase 2)")
    parser.add_argument("--phase1-epochs", type=int, default=3, help="Epochs with frozen backbone")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for phase 2")
    parser.add_argument("--phase1-lr", type=float, default=1e-3, help="Learning rate for phase 1 (head only)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (val acc)")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_dataset = ImageFolder(train_dir, transform=build_transforms(train=True, image_size=args.image_size))
    val_dataset = ImageFolder(val_dir, transform=build_transforms(train=False, image_size=args.image_size))
    class_names = train_dataset.classes
    print(f"Classes détectées : {class_names}")

    # --- Poids de classes en cas de déséquilibre --------------------------------
    counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=len(class_names))
    class_weights = (1.0 / counts.float())
    class_weights = class_weights / class_weights.sum()  # normalisé
    print(f"Distribution train : {dict(zip(class_names, counts.tolist()))}")
    print(f"Poids par classe  : {dict(zip(class_names, class_weights.tolist()))}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_model(num_classes=len(class_names), pretrained=not args.no_pretrained).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(args.device))

    # --- Boucle d'entraînement en deux phases -----------------------------------
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    writer = SummaryWriter(log_dir="runs/spoof_finetune")
    optimizer = None
    scheduler = None

    for epoch in range(1, args.epochs + 1):
        phase_name, freeze = phase_hyperparams(epoch, args.phase1_epochs, args.epochs - args.phase1_epochs)

        # Applique le gel/dégel au début de la phase 1 (et pour re-geler si besoin)
        # On gèle au tout premier epoch et on dégèle au premier epoch de phase 2.
        requires_freeze = freeze
        if epoch == args.phase1_epochs + 1:
            requires_freeze = False

        if epoch == 1 or epoch == args.phase1_epochs + 1:
            for name, param in model.named_parameters():
                param.requires_grad = not requires_freeze
            # On recrée un optimiseur propre à chaque changement de phase
            if phase_name == "phase1":
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.phase1_lr)
            else:
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.phase1_epochs + 1)
            print(f"\n[{phase_name.upper()}] {'Backbone GELÉ' if requires_freeze else 'Backbone DÉGELÉ (full fine-tune)'} | lr={optimizer.param_groups[0]['lr']}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, device=args.device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=args.device)
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch:2d}/{args.epochs} [{phase_name:6s}] | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        writer.add_scalars(
            "loss", {"train": train_loss, "val": val_loss}, epoch
        )
        writer.add_scalars(
            "accuracy", {"train": train_acc, "val": val_acc}, epoch
        )

        # Mise à jour du meilleur modèle + early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Pas d'amélioration depuis {epochs_no_improve} epoch(s) (best_val_acc={best_val_acc:.4f})")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping déclenché après {epoch} epochs.")
                break

    writer.close()

    # --- Sauvegarde du meilleur modèle ------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    save_checkpoint(model, class_names, args.output)
    print(f"\nMeilleure val accuracy : {best_val_acc:.4f}")
    print(f"Checkpoint sauvegardé : {args.output}")

    # --- Évaluation finale détaillée ---------------------------------------------
    val_acc, report, cm = evaluate_model(model, val_loader, class_names, device=args.device)
    print("\n=== Évaluation finale sur la validation ===")
    print(f"Accuracy : {val_acc:.4f}")
    print("\nClassification report :")
    print(report)
    print("Matrice de confusion :")
    print(cm)


if __name__ == "__main__":
    main()