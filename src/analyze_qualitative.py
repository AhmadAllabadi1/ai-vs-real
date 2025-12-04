import os
import shutil
import torch
from dataset import get_dataloaders
from model import CNNViTHybrid  # make sure this import path is correct

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path="model.pth", num_classes=2):
    model = CNNViTHybrid(num_classes=num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_examples(records, out_dir, k=30):
    os.makedirs(out_dir, exist_ok=True)
    for r in records[:k]:
        fname = os.path.basename(r["path"])
        tagged_name = f"{r['p_real']:.3f}real_{r['p_fake']:.3f}fake_{fname}"
        dst = os.path.join(out_dir, tagged_name)
        shutil.copy(r["path"], dst)


if __name__ == "__main__":
    # 1) Get dataloaders (uses your eval_transform for test)
    _, _, test_loader, num_classes = get_dataloaders(
        batch_size=64,
        num_workers=4,
    )

    test_dataset = test_loader.dataset  # this is ImageFolder(data/test, ...)
    print("class_to_idx:", test_dataset.class_to_idx)
    # With folders data/test/fake and data/test/real this should be:
    # {'fake': 0, 'real': 1}
    fake_idx = test_dataset.class_to_idx["fake"]
    real_idx = test_dataset.class_to_idx["real"]

    # 2) Load trained model
    model = load_model("model.pth", num_classes=num_classes)

    # 3) Run inference over the whole test set and collect records
    records = []
    global_idx = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            for i in range(images.size(0)):
                path, _ = test_dataset.samples[global_idx]
                p_fake = probs[i, fake_idx].item()
                p_real = probs[i, real_idx].item()

                records.append({
                    "path": path,
                    "true": labels[i].item(),   # 0 = fake, 1 = real (if folders are fake/real)
                    "pred": preds[i].item(),
                    "p_fake": p_fake,
                    "p_real": p_real,
                })
                global_idx += 1

    # 4) Filter into the three qualitative buckets you described

    # a) Obvious artifacts: AI fake correctly classified as fake (high p_fake)
    correct_fake = [
        r for r in records
        if r["true"] == fake_idx and r["pred"] == fake_idx
    ]
    correct_fake = sorted(correct_fake, key=lambda r: r["p_fake"], reverse=True)
    save_examples(correct_fake, "qualitative/correct_fake_obvious", k=30)

    # b) Highly realistic AI misclassified as real: true=fake, pred=real
    mis_ai_as_real = [
        r for r in records
        if r["true"] == fake_idx and r["pred"] == real_idx
    ]
    mis_ai_as_real = sorted(mis_ai_as_real, key=lambda r: r["p_real"], reverse=True)
    save_examples(mis_ai_as_real, "qualitative/ai_misclassified_as_real", k=30)

    # c) Real images misclassified as fake: true=real, pred=fake
    mis_real_as_fake = [
        r for r in records
        if r["true"] == real_idx and r["pred"] == fake_idx
    ]
    mis_real_as_fake = sorted(mis_real_as_fake, key=lambda r: r["p_fake"], reverse=True)
    save_examples(mis_real_as_fake, "qualitative/real_misclassified_as_fake", k=30)

    print("Saved qualitative examples under ./qualitative/**")
