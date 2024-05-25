import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F 
import os

class ContrastiveLoss(torch.nn.Module):
    """Class that inherit from Base class for all neural network modules (torch.nn.Module) that 
    will have the loss function for the model.
    
    Contrastive Loss Function
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
      
      return loss_contrastive

class SiameseNetwork(nn.Module):
    """
    The Siamese Neural Network architecture based on Siamese Neural Networks for One-shot Image Recognition paper
    but with some changes to have better results in face verification.
    """
    
    counter = []
    loss_history = []
    
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256,2)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    
    def show_plot(self):
        # The function that plot the stats of the model during training
        plt.plot(self.counter, self.loss_history)
        plt.show()
    
    def train(self, Epochs, data, LearningRate):
        """The training loop for the model.

        Args:
            Epochs (int): Number of Epochs for the training loop.
            data (tensor): The images that the model will train on them.
            LearningRate (double): The learning rate of the model that will be multiplied with the weights.
        """
        # Using the loss function created before
        criterion = ContrastiveLoss()
        # Using Adam optimizer for the model
        optimizer = optim.Adam(self.parameters(), lr = LearningRate)
        
        iteration_number= 0

        # Iterate throught the epochs
        for epoch in range(Epochs):

            # Iterate over batches
            for i, (img0, img1, label) in enumerate(data, 0):

                # Send the images and labels to CUDA
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

                # Zero the gradients
                optimizer.zero_grad()

                # Pass in the two images into the network and obtain two outputs
                output1, output2 = self(img0.to(torch.device('cuda')), img1.to(torch.device('cuda')))

                # Pass the outputs of the networks and label into the loss function
                loss_contrastive = criterion(output1, output2, label.to(torch.device('cuda')))

                # Calculate the backpropagation
                loss_contrastive.backward()

                # Optimize
                optimizer.step()

                # Every 10 batches print out the loss
                if i % 10 == 0 :
                    print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                    iteration_number += 10

                    self.counter.append(iteration_number)
                    self.loss_history.append(loss_contrastive.item())
        
    def predict(self, data):
        result = []
        # Grab one image that we are going to test
        dataiter = iter(data)
        x0, _ = next(dataiter)
        # Iterate through all of the verification images folder to compare the input image with verification images
        for i in range(len(os.listdir(r"application_data\input_image\verification_images")) - 1):
            _ , x1 = next(dataiter)
            
            output1, output2 = self(x0.cuda(), x1.cuda())
            euclidean_distance = F.pairwise_distance(output1, output2)
            result.append(euclidean_distance.item())

        return result