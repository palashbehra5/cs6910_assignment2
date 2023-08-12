from data import train_loader,val_loader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models import baseline, resnet, vit, densenet, inception, se_resnet, convnext, CAM, CAM_resnet, student
import torch

def criterion(outputs_student, outputs_teacher, outputs_gt):
     
    criterion_training = nn.CrossEntropyLoss(reduction='mean')
    criterion_distance = nn.KLDivLoss(reduction='batchmean')

    probs_teacher = F.softmax(outputs_teacher, dim=1)
    probs_student = F.softmax(outputs_student, dim=1)

    Loss = 0.8 * criterion_distance(probs_student.log(), probs_teacher) + 0.2 * criterion_training(outputs_student, outputs_gt)
    return Loss

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
PATH = "models\\models_petimages\\distillation.pth"
TEACHER_PATH = "models\\models_petimages\\densenet.pth"

teacher = densenet(num_labels=2).to('cuda')
stud = student(num_labels=2).to('cuda')

checkpoint = torch.load(TEACHER_PATH)
teacher.load_state_dict(checkpoint)
teacher.eval()

for param in teacher.parameters():
        param.requires_grad = False

optimizer = optim.Adam(stud.parameters(), 
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
num_epochs = 50
max_val_acc = 0

for epoch in range(num_epochs):

    stud.train()  
    total_loss = 0.0

    for inputs, labels in train_loader:

        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        outputs_student = stud(inputs)
        outputs_teacher = teacher(inputs)
        outputs_gt = labels

        loss = criterion(outputs_student, outputs_teacher, outputs_gt)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    stud.eval()  
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            outputs_student = stud(inputs)
            _, predicted = torch.max(outputs_student, 1)

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)


    val_accuracy = correct_predictions / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    if(val_accuracy > max_val_acc):

        max_val_acc = val_accuracy
        torch.save(stud.state_dict(), PATH)
        print("Checkpointing--{}".format(max_val_acc))

print("Training complete!")