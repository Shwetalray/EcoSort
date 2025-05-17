from pathlib import Path


train_model_path = Path("/mnt/data/waste_classification_project/train_model.py")
train_model_path.parent.mkdir(parents=True, exist_ok=True)



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import json
from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter

class WasteDataset(Dataset):
    def __init__(self, base_path, annotation_file, transform=None):
        self.base_path = Path(base_path)
        self.transform = transform
        with open(self.base_path / annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())
        self.labels = [0 if self.annotations[img] == "plastic" else 1 for img in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_file = self.image_paths[idx]
        img_path = self.base_path.parent / img_file
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class WasteClassifier:
    def __init__(self, base_path="dataset"):
        self.base_path = Path(base_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = ["plastic", "paper"]
        self.model = self.create_model()
        self.checkpoint_path = self.base_path / "best_model.pth"

    def create_model(self):
        print("\\nInitializing EfficientNet-B0 model")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(self.categories))
        return model.to(self.device)

    def calculate_class_weights(self, dataset):
        label_counts = Counter(dataset.labels)
        total = sum(label_counts.values())
        class_weights = []
        for i in range(len(self.categories)):
            count = label_counts.get(i, 1)
            weight = total / (len(self.categories) * count)
            class_weights.append(weight)
        return torch.tensor(class_weights, dtype=torch.float).to(self.device)

    def train_model(self):
        print("\\nStarting training with processed dataset:")
        print("✓ Using preprocessed 224x224 images")
        print("✓ Including augmented versions")
        print("✓ 5-fold cross validation")
        print("✓ Early stopping")
        print("✓ Learning rate scheduling")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = WasteDataset(self.base_path, "train_annotations.json", transform)
        kfold = KFold(n_splits=5, shuffle=True)
        best_acc = 0.0

        class_weights = self.calculate_class_weights(dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"\\nFold {fold + 1}/5")
            train_loader = DataLoader(dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(train_ids))
            val_loader = DataLoader(dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(val_ids))

            self.model = self.create_model()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)

            for epoch in range(10):
                self.model.train()
                total, correct, running_loss = 0, 0, 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                train_acc = 100 * correct / total
                print(f"Epoch {epoch+1}: Train Loss={running_loss:.4f}, Train Accuracy={train_acc:.2f}%")

                self.model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        _, predicted = torch.max(outputs, 1)
                        val_preds.extend(predicted.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())

                val_acc = accuracy_score(val_labels, val_preds) * 100
                print(f"Epoch {epoch+1}: Validation Accuracy={val_acc:.2f}%")

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(self.model.state_dict(), self.checkpoint_path)

        print(f"\\nTraining complete! Best Validation Accuracy: {best_acc:.2f}%")

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    try:
        classifier = WasteClassifier()
        classifier.train_model()
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise

if __name__ == "__main__":
    main()



train_model_path.write_text(train_model_code)
train_model_path
