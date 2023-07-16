""""Inference with Trained Model on Single Image"""

from PIL import Image
import torch
import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt

IMAGE_INPUT_SIZE = (28, 28)


device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = "6.png"
image = Image.open(image_path).convert('L')

model = torch.load('modelS/model_after_epoch_19.pt').to(device)

#Set the model to evaluation mode
model.eval()

preprocess = transforms.Compose([transforms.Resize(IMAGE_INPUT_SIZE), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

input_tensor = preprocess(image)
input_tensor = input_tensor.to(device)
input_tensor = torch.unsqueeze(input_tensor, 0)

output = model(input_tensor)
output = nn.functional.softmax(output, 1)
predicted_class = output.argmax(1).item()  

#Retrieve the probability of the predicted class
probability = output[0][predicted_class]

# Display the input image and prediction information
plt.imshow(image, cmap='gray')
plt.title('Class: {}\nProbability: {:.4f}'.format(predicted_class, probability))
plt.show()