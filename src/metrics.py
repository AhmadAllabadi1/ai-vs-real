import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def eval_with_metrics(model, loader, device):
    model.eval()

    all_labels = []
    all_probs = []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_preds = np.argmax(all_probs, axis=1)

    acc = correct / total
    prec = precision_score(all_labels, all_preds, average="binary")
    rec = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)

    return acc, prec, rec, f1, cm, all_labels, all_probs
