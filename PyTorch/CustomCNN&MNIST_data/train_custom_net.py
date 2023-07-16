import torch
import math
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from my_conv_nn import MyConvNeuralNetwork

BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_OF_EPOCHS = 20
ITERATION_LOGGING_FREQUENCY = 100
EXPERIMENT_NAME = "runs1/custom_net_mnist2"
MODELS_PATH = "modelS"

overall_iteration = 0
writer = SummaryWriter(EXPERIMENT_NAME) # 

device = "cuda" if torch.cuda.is_available() else 'cpu' 


preproces = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])]) 

mnist_dataset_train = MNIST('data', train=True, download=True, transform=preproces) 
mnist_dataset_validation = MNIST('data', train=False, download=True, transform=preproces)


fig, axes = plt.subplots(3) 
for i in range(3):
    image, label = mnist_dataset_train[i]
    image = image.squeeze() 
    axes[i].imshow(image, cmap = 'gray')
    axes[i].set_title(str(label))
plt.show()

mnist_dataloader_train = DataLoader(mnist_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
mnist_dataloader_validation = DataLoader(mnist_dataset_validation, batch_size=BATCH_SIZE)

loss_fn = nn.CrossEntropyLoss()

model = MyConvNeuralNetwork().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)  

def train(dataloader, model, loss_fn, optimizer, epoch):
    running_loss = 0
    dataset_size = len(dataloader.dataset)
    for iteration_in_epoch, (images, labels) in enumerate(dataloader):
        global overall_iteration
        images, labels = images.to(device), labels.to(device) 
        output = model(images)

        loss = loss_fn(output, labels) 
        loss.backward()
        optimizer.step() 

        running_loss += loss.item()
        overall_iteration += 1

        if (iteration_in_epoch + 1) % ITERATION_LOGGING_FREQUENCY == 0:
            average_loss = running_loss / ITERATION_LOGGING_FREQUENCY
            learning_rate = optimizer.param_groups[0]['lr'] 
            writer.add_scalar("training_loss", average_loss, overall_iteration)
            writer.add_scalar("learning_rate", learning_rate, overall_iteration)
            running_loss = 0
            print("Epoch {} / {} [{} / {}] Loss: {:.4f} learning rate: {}".format(epoch, 
                                                                                  NUM_OF_EPOCHS, 
                                                                                  iteration_in_epoch+1, 
                                                                                  math.ceil(dataset_size//BATCH_SIZE),
                                                                                  average_loss, 
                                                                                  learning_rate))
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
        print("Validation accuracy {:.2f}".format(accuracy))
        torch.save(model, MODELS_PATH + "/model_after_epoch_{}.pt".format(epoch_num))  
    print("Done with validation!")


for epoch in range(1, NUM_OF_EPOCHS + 1): 
    train(mnist_dataloader_train, model, loss_fn, optimizer, epoch)
    validate(mnist_dataloader_validation, model, epoch)
