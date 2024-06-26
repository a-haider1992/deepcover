import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50, resnet18, ResNet50_Weights, vgg16, inception_v3
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from dataset import CustomImageDataset, ExplanationsPatchesDataset
import torchvision.models as models
from sklearn.metrics import confusion_matrix , accuracy_score, classification_report, f1_score
import pdb

# Define the ResNet-50 model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.model = resnet50(weights='DEFAULT')

        # Optionally unfreeze some layers
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for param in self.model.layer4.parameters():
        #     param.requires_grad = True

        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(256, num_classes),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# CLAHE
class CLAHETransform:
    def __init__(self, clip_limit=5.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        # Convert Tensor to PIL image if it's a tensor
        if isinstance(img, torch.Tensor):
            img = self.to_pil(img)

        # Convert PIL image to numpy array
        img = np.array(img)
        if len(img.shape) == 2:  # If the image is grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply CLAHE to each channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Convert numpy array back to PIL image
        img = Image.fromarray(img)
        # Convert PIL image back to Tensor
        # img = self.to_tensor(img)
        return img


# Initialize the model
num_classes = 3
num_epochs = 200

# ResNet
model = ResNetClassifier(num_classes)

# VGG16
# vgg16 = models.vgg16(pretrained=True)

# pdb.set_trace()
# image_test = Image.open('image1.jpg')
# image_test_clahe = CLAHETransform()(image_test)
# image_test_clahe.show()

image_size = 256  # Resize to 256x256
crop_size = 224  # Center crop to 224x224

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize([image_size, image_size]),
    transforms.CenterCrop(crop_size),  # Apply center crop
    CLAHETransform(clip_limit=1.0, tile_grid_size=(8, 8)),  # Apply CLAHE
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create a custom dataset and dataloader
dataset = CustomImageDataset(txt_file="fundus_annotate_train.txt", root_dir=".", transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
# test_dataset = datasets.ImageFolder(root='../data/Fundus', transform=transform)
test_dataset = CustomImageDataset(txt_file="fundus_annotate_test.txt", root_dir=".", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Fine-tune the model
criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)


best_f1 = 0.0
early_stopping_patience = 10
no_improvement_epochs = 0

for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()
    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    # scheduler.step()
    epoch_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch {epoch + 1}/{num_epochs}, F1-Score: {f1}")

    if f1 > best_f1:
        best_f1 = f1
        no_improvement_epochs = 0
        torch.save(model.state_dict(), "best_fundus_classifier.pth")
    else:
        no_improvement_epochs += 1

    if no_improvement_epochs >= early_stopping_patience:
        print("Early stopping due to no improvement")
        break

print(f"Best F1-Score: {best_f1}")

# # Evaluate the model
# model.load_state_dict(torch.load("best_fundus_classifier.pth"))
# correct = 0
# total = 0
# accuracy = 0.0
# model.eval()
# true_labels = []
# predicted_labels = []
# # pdb.set_trace()
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         images, labels = inputs.to(device), labels
#         outputs = model(images)
#         # pdb.set_trace()
#         _, predicted = torch.max(outputs.data, 1)
#         total += len(labels)
#         correct = 0
#         for i in range(len(labels)):
#             true_labels.append(labels[i].item())
#             predicted_labels.append(predicted[i].item())
#             if labels[i] == predicted[i]:
#                 correct += 1
#         accuracy += correct / total
#         # print(f'correct: {correct}, total: {total}')
#     accuracy /= len(test_loader)
#     print(f"Test Accuracy: {accuracy * 100:.2f}%")
#     print(f"Confusion Matrix: {confusion_matrix(true_labels, predicted_labels)}")
#     print(f"Classification Report: {classification_report(true_labels, predicted_labels)}")
#     # write the classification report to a file
#     with open('classification_report.txt', 'w') as f:
#         f.write(classification_report(true_labels, predicted_labels))
#     with open('confusion_matrix.txt', 'w') as f:
#         f.write(str(confusion_matrix(true_labels, predicted_labels)))
