import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os
from tqdm import tqdm
from PIL import Image
import timm
from dataset import CustomImageDataset

class Trainer:
    def __init__(self, save_path):
        """
        Initialize the Trainer with data transformations, datasets, model, and optimization components.
        
        Args:
        save_path (str): Directory to save model checkpoints.
        """
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
            transforms.ToTensor(),  # Convert images to PyTorch tensors (scales values to [0, 1])
            transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5 and std=0.5 (scales to [-1, 1])
        ])
        
        self.save_path = save_path
        
        # Create datasets and data loaders
        self.train_dataset = CustomImageDataset(root_dir='./custom_train_images', transform=self.transform)
        self.test_dataset = CustomImageDataset(root_dir='./custom_test_images', transform=self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=10, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=10, shuffle=False)
        
        # Initialize the model
        # Initialize the VGG-19 model
        self.model = torchvision.models.vgg19(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, 6)  # Modify the final layer for 6 classes
        self.criterion = nn.CrossEntropyLoss()

        # Define optimizer with learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # Move model to the selected device
        self.class_names = self.test_dataset.classes
        print(self.class_names)

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs.
        
        Args:
        num_epochs (int): Number of epochs to train for.
        """
        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode (enables dropout, batch norm updates, etc.)
            running_loss = 0.0
            
            # Training loop
            for images, labels in tqdm(self.train_loader):
                # Move data to the selected device
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()  # Zero the parameter gradients
                
                outputs = self.model(images)  # Forward pass: compute predicted outputs by passing inputs to the model
                
                # Calculate the loss
                # CrossEntropyLoss expects raw outputs and class indices, not one-hot encoded
                # It internally applies softmax to the output
                loss = self.criterion(outputs, labels)
                
                loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
                self.optimizer.step()  # Perform a single optimization step (parameter update)
                
                running_loss += loss.item()  # Accumulate the loss
            
            # Print average loss for the epoch
            #print(f"Epoch {epoch+1}, Loss: {running_loss / len(self.train_loader)}")
            
            # Evaluation loop
            self.model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm, etc.)
            correct = 0
            total = 0
            with torch.no_grad():  # Disable gradient computation for efficiency during evaluation
                for images, labels in tqdm(self.test_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    predicted = torch.argmax(outputs.data, 1)  # Get the index of the max log-probability
                    predicted_labels = torch.argmax(labels.data, 1)  # Convert one-hot to class indices
                    predicted_class_names = [self.class_names[idx] for idx in predicted]
                    total += labels.size(0)
                    correct += (predicted == predicted_labels).sum().item()
            
            # print(predicted_class_names)
            # Print accuracy
            print(f"Accuracy on test set: {(correct / total) * 100}%")
            
            # Save model checkpoint
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.model, save_path + os.sep + str(epoch) + ".pt")
            # Within the train method, update the save checkpoint path
            # torch.save(self.model, f"{save_path}/vgg19_emotion_classification_{epoch}.pt")

# Usage
save_path = "vgg19_model_checkpoints"
trainer = Trainer(save_path)
trainer.train(1000)