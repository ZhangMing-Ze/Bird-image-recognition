import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

# 参数配置
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005
NUM_CLASSES = 71
SAMPLE_RATE = 44100
DURATION = 10
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 224

# ----------------------
# 1. 数据预处理与加载
# ----------------------
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            audio, _ = librosa.load(self.file_paths[idx],
                                     sr=SAMPLE_RATE,
                                     duration=DURATION)
            if len(audio) < SAMPLE_RATE * DURATION:
                audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)), mode='constant')
            else:
                audio = audio[:SAMPLE_RATE * DURATION]

            mel_spec = librosa.feature.melspectrogram(y=audio,
                                                       sr=SAMPLE_RATE,
                                                       n_fft=N_FFT,
                                                       hop_length=HOP_LENGTH,
                                                       n_mels=N_MELS)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            norm_spec = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            image = np.stack([norm_spec] * 3, axis=-1)

            if self.transform:
                image = self.transform(image)

            label = self.labels[idx] if len(self.labels) > 0 else 0
            return image, label

        except Exception as e:
            print(f"Error processing {self.file_paths[idx]}: {str(e)}")
            return torch.zeros(3, 224, 224), 0


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def prepare_data(data_dir):
    file_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))  # 排序确保顺序一致

    for label, bird_class in enumerate(class_names):
        class_dir = os.path.join(data_dir, bird_class)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith((".wav", ".mp3")):
                    file_paths.append(os.path.join(class_dir, file_name))
                    labels.append(label)

    return file_paths, labels, class_names


# ----------------------
# 2. 模型构建
# ----------------------
def create_model(num_classes):
    try:
        weights = models.ResNet34_Weights.IMAGENET1K_V1
    except AttributeError:
        weights = None

    model = models.resnet34(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(num_features, num_classes)
    )
    return model


# ----------------------
# 3. 训练流程 + 可视化 + 混淆矩阵
# ----------------------
def main():
    data_dir = r"D:\鸟类总数据集"  # 修改为你的路径
    file_list, label_list, class_names = prepare_data(data_dir)

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_list, label_list, test_size=0.2, stratify=label_list)

    train_dataset = AudioDataset(train_files, train_labels, transform=transform)
    val_dataset = AudioDataset(val_files, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_acc = 0.0
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss = val_loss / len(val_dataset)
        val_acc = correct / len(val_dataset)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model1.pth")
            print(f"Saved new best model with accuracy: {val_acc:.4f}")

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}\n")

    # --- 可视化训练过程 ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_visualization.png")
    plt.show()

    # --- 混淆矩阵 ---
    all_preds = []
    all_labels = []

    model.load_state_dict(torch.load("best_model1.pth"))
    model.eval()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == '__main__':
    freeze_support()
    main()
