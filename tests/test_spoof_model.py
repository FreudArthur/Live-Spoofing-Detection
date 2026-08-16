import tempfile
import unittest

import torch
from PIL import Image

from spoof_model import create_model, load_checkpoint, predict_pil_image, save_checkpoint


class SpoofModelTests(unittest.TestCase):
    def test_create_model_output_size(self):
        model = create_model(num_classes=2, pretrained=False)
        self.assertEqual(model.fc.out_features, 2)

    def test_predict_pil_image_returns_probabilities(self):
        model = create_model(num_classes=2, pretrained=False)
        image = Image.new("RGB", (224, 224), color="white")
        class_names = ["non-spoof", "spoof"]

        label, score, class_scores = predict_pil_image(model, image, class_names)

        self.assertIn(label, class_names)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(set(class_scores.keys()), set(class_names))
        self.assertAlmostEqual(sum(class_scores.values()), 1.0, places=5)

    def test_checkpoint_roundtrip(self):
        model = create_model(num_classes=2, pretrained=False)
        class_names = ["non-spoof", "spoof"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = f"{tmp_dir}/model.pt"
            save_checkpoint(model, class_names, ckpt_path)
            loaded_model, loaded_classes = load_checkpoint(ckpt_path, device="cpu")

        self.assertEqual(loaded_classes, class_names)
        self.assertIsNotNone(loaded_model)
        self.assertIsInstance(next(loaded_model.parameters()), torch.Tensor)


if __name__ == "__main__":
    unittest.main()
