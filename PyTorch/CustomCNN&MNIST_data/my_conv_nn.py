""" Custom Convolutional Neural Network (CNN) for Image Classification"""

from torch import nn 
import torch.nn.functional as F  #activation functions

#Model Definition
class MyConvNeuralNetwork(nn.Module): 
    def __init__(self):
        super().__init__()   

        self.__conv1 = nn.Conv2d(1, 10, (5,5), padding = 2) 
        self.__conv2 = nn.Conv2d(10, 20, (5,5), padding = 2) 

        self.__pool = nn.MaxPool2d(2)

        self.__fc1 = nn.Linear(980, 50)
        self.__fc2 = nn.Linear(50, 10)
    #Forward pass
    def forward(self, x):
        x = self.__conv1(x)
        x = F.relu(x)
        x = self.__pool(x)

        x = self.__conv2(x)
        x = F.relu(x)
        x = self.__pool(x)

        x = x.view(-1, 980) 

        x = self.__fc1(x)
        x = F.relu(x)
        x = self.__fc2(x)

        return x  
