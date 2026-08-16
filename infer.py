import argparse

from PIL import Image

from spoof_model import load_checkpoint, predict_pil_image


def main():
    parser = argparse.ArgumentParser(description="Run spoof/non-spoof inference on one image")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, class_names = load_checkpoint(args.checkpoint, device=args.device)
    image = Image.open(args.image).convert("RGB")
    label, score, class_scores = predict_pil_image(
        model, image, class_names, device=args.device, image_size=args.image_size
    )

    print(f"Prediction: {label} ({score:.4f})")
    for class_name, class_score in class_scores.items():
        print(f"{class_name}: {class_score:.4f}")


if __name__ == "__main__":
    main()
