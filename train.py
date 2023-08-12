from data import train_loader,val_loader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models import baseline, resnet, vit, densenet, inception, se_resnet, convnext, CAM, CAM_resnet
import torch

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
PATH = "models\\models_petimages\\CAM_resnet.pth"

model = CAM_resnet(num_labels=2).to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_epochs = 20
max_val_acc = 0

for epoch in range(num_epochs):

    model.train()  
    total_loss = 0.0

    for inputs, labels in train_loader:

        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    model.eval()  
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)


    val_accuracy = correct_predictions / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    if(val_accuracy > max_val_acc):

        max_val_acc = val_accuracy
        torch.save(model.state_dict(), PATH)
        print("Checkpointing--{}".format(max_val_acc))

print("Training complete!")