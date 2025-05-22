from sklearn.metrics._plot.confusion_matrix import confusion_matrix
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from dataset_loader import ProteinDatasetLoader, ProteinDataset
from model import DeepLocModel


def train_model(model, train_loader, epochs=20, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")


def evaluate(model, test_loader, device="cpu"):
    model.eval()
    val_loss = 0.0
    criterion = nn.BCELoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            preds = (outputs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    avg_val_loss = val_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)

    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


def plot_confusion_matrix(model, test_loader, device="cpu", class_names=None):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mcm = multilabel_confusion_matrix(all_targets, all_preds)

    if class_names is None:
        class_names = [
            "Cell membrane", "Cytoplasm", "Endoplasmic reticulum", "Golgi apparatus",
            "Lysosome/Vacuole", "Mitochondrion", "Nucleus", "Peroxisome"
        ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, cm in enumerate(mcm):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Not {class_names[i]}', class_names[i]])
        disp.plot(ax=axes[i], colorbar=False, cmap='Blues', values_format='d')
        axes[i].set_title(class_names[i])

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")


def main():
    # Load the datasets
    train_df = ProteinDatasetLoader(
        "data/Swissprot_Train_Validation_dataset.csv"
    ).load()
    test_df = ProteinDatasetLoader("data/hpa_testset.csv").load()

    # Train the model
    model = DeepLocModel()
    train_loader = DataLoader(ProteinDataset(train_df), batch_size=32, shuffle=True)
    test_loader = DataLoader(ProteinDataset(test_df), batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, epochs=5, lr=1e-3, device=device)
    evaluate(model, test_loader, device=device)

    # Confusion matrix
    plot_confusion_matrix(model, test_loader, device=device)


if __name__ == "__main__":
    main()
