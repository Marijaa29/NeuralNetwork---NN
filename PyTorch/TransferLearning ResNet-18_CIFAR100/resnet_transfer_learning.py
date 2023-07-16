""""Transfer Learning with CIFAR-100 and ResNet-18: Training and Validation with TensorBoard Logging"""

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR100
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import math
from torchvision import models 



BATCH_SIZE = 8
LEARNING_RATE = 0.01
NUM_OF_EPOCHS = 20
ITERATION_LOGGING_FREQUENCY = 100
EXPERIMENT_NAME = "runs/initial_experiment_lr_0-01"
MODELS_PATH = "models"

overall_iteration = 0
writer = SummaryWriter(EXPERIMENT_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu" 

#Transformation of data (preprocessing)
preprocess = transforms.Compose([transforms.Resize(224), #give a list of transformations
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                 ])  


#Dataset for pytorch 
cifar_dataset_train = CIFAR100('data', train=True, download=True, transform=preprocess) 
cifar_dataset_validation = CIFAR100('data', train=False, download=True, transform=preprocess)

class_names = cifar_dataset_train.classes

print(class_names)

fig, axes = plt.subplots(3)

for i in range(3):
    image, label = cifar_dataset_train[i]
    image = torch.transpose(image, 0,1)
    image = torch.transpose(image, 1,2)
    axes[i].imshow(image)
    axes[i].set_title(class_names[label])
plt.show()


#Define dataloader
cifar_dataloader_train = DataLoader(cifar_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
cifar_dataloader_validation = DataLoader(cifar_dataset_validation, batch_size=BATCH_SIZE)    

#Define loss function
loss_fn = nn.CrossEntropyLoss()

#Model
model = models.resnet18(weights='DEFAULT')
fc_layer_in_features = model.fc.in_features
model.fc = nn.Linear(fc_layer_in_features, 100)
model = model.to(device)

#Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

#Training Function
def train(dataloader, model, loss_fn, optimizer, epoch):
    global overall_iteration
    running_loss = 0
    dataset_size = len(dataloader.dataset)
    for iteration_in_epoch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        overall_iteration += 1                 
        
        if (iteration_in_epoch + 1) % ITERATION_LOGGING_FREQUENCY == 0:
            average_loss = running_loss / ITERATION_LOGGING_FREQUENCY
            learning_rate = optimizer.param_groups[0]['lr']
            writer.add_scalar("training loss", average_loss, overall_iteration)
            writer.add_scalar("learning rate", learning_rate, overall_iteration)
            running_loss = 0
            print("Epoch {} / {} [{}/{}] Loss: {:.4f} Learning rate: {}".format(epoch, NUM_OF_EPOCHS, iteration_in_epoch+1, math.ceil(dataset_size / BATCH_SIZE), average_loss, learning_rate))
 
#Validation Function            
def validate(dataloader, model, epoch_num):
    print("Start validating...")
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            predictions= nn.functional.softmax(predictions, dim=1)
            predicted_classes = predictions.argmax(dim=1)
            correct_predicted_classes = (predicted_classes == labels).sum().item()
            total_correct += correct_predicted_classes
        accuracy = total_correct / len(dataloader.dataset)
        writer.add_scalar("validation_accuracy", accuracy, epoch_num)
        print("Validation accuracy: {:.2f}".format(accuracy))
        torch.save(model, MODELS_PATH + "/model_after_epoch_{}.pt".format(epoch_num))
    print("Done with validation!")

#Main Training Loop
for epoch in range(1, NUM_OF_EPOCHS+1):
    train(cifar_dataloader_train, model, loss_fn, optimizer, epoch)
    validate(cifar_dataloader_validation, model, epoch)