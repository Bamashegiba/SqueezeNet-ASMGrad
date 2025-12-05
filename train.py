# train.py
import torch
from torch import nn, optim
from torchvision import models
import time
from tqdm import tqdm
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from torch.optim.lr_scheduler import OneCycleLR

from dataset import get_loaders
from model import SqueezeNet





def validate(model, val_loader, device):

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            total_loss += loss.item()

            pred_class = preds.argmax(dim=1)
            correct += (pred_class == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / len(val_loader)
    acc = correct / total

    return avg_loss, acc


def test(model, test_loader, device):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            pred_class = preds.argmax(dim=1)
            correct += (pred_class == y).sum().item()
            total += y.size(0)

    return correct / total


def train_model(
        pretrained,
        model_path,
        epochs=10,
        lr=1e-4,
        weight_decay=1e-4,
        use_amsgrad=False
):

    DATA_ROOT = "data/Dataset/Images"
    start_time = time.perf_counter()
    history = {
        "lr_val": [],
        "loss_val": [],
        "accuracy_val": [],
        "precision_val": [],
        "recall_val": []
    }

    train_loader, val_loader, test_loader, class_names = get_loaders(
        DATA_ROOT,
        batch_size=32,
        img_size=224,
        num_workers=4
    )
    num_classes = len(class_names)

    if pretrained:
        model = models.squeezenet1_1(weights="IMAGENET1K_V1" if pretrained else None)
        # Заменяем последний слой
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes
    else:
        model = SqueezeNet(out_channels=num_classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=use_amsgrad
    )

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=3
    # )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr * 10,  # максимальный LR в цикле (обычно 3-10x от начального)
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        anneal_strategy='cos'  # cosine annealing
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_loss, val_acc = validate(model, val_loader, device)
        all_preds = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                preds = model(vx).argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(vy.cpu().tolist())
        # scheduler.step()

        precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        current_lr = optimizer.param_groups[0]["lr"]


        history["lr_val"].append(current_lr)
        history["loss_val"].append(val_loss)
        history["accuracy_val"].append(val_acc)
        history["precision_val"].append(precision)
        history["recall_val"].append(recall)


        print(f"\nEpoch {epoch}/{epochs} | "
              f"train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | "
              f"val accuracy: {val_acc:.4f} | "
              f"val precision: {precision:.4f} | "
              f"val recall: {recall:.4f} | "
              f"val LR: {current_lr:.8f}")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()


    if best_state:
        model.load_state_dict(best_state)

    result_time = time.perf_counter() - start_time
    test_acc = test(model, test_loader, device)
    print(f"\nFinal TEST accuracy: {test_acc:.4f}; Time spent:{result_time/60:.2f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    csv_name = "CSV/" + model_path[7:-4] + "_training_history.csv"
    df = pd.DataFrame(history)
    df.to_csv(csv_name, index=False)
    print("Logged in ", csv_name)

    return model, test_acc


