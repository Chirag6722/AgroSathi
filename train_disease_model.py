import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import json
import os

def train(data_dir='datasets/disease', epochs=3, batch=16, save_path='models/disease_model.pth', labels_path='models/disease_labels.json'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Data transforms
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # dataset & dataloader
    train_ds = datasets.FakeData(transform=train_tf, size=500, num_classes=7, image_size=(3,224,224))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True)

    # model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("✅ Model saved to:", save_path)

    # save label mapping
    label_map = {str(i): cls for i, cls in enumerate([
        "Healthy",
        "Late Blight",
        "Powdery Mildew",
        "Leaf Spot",
        "Rust",
        "Bacterial Spot",
        "Downy Mildew"
    ])}
    with open(labels_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    print("✅ Labels saved to:", labels_path)

if __name__ == '__main__':
    train()
