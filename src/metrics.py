from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch

def eval_with_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = sum(p == y for p, y in zip(all_preds, all_labels)) / len(all_labels)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    cm = confusion_matrix(all_labels, all_preds)
    return acc, prec, rec, f1, cm
