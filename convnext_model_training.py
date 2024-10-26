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

from matplotlib import pyplot as plt

from ultralytics import YOLO
from torchvision.transforms import functional as F

class CropFaceTransform:
    def __init__(self, face_model):
        self.face_model = face_model

    def __call__(self, image):
        # Perform face detection using the provided YOLO face detection model
        result = self.face_model(image)
        boxes = result[0].boxes.xyxy.cpu().numpy()

        # Extract the bounding box of the face and crop the face image
        if len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box[:4])
            face_image = F.crop(image, y1, x1, y2-y1, x2-x1)
            # plt.imshow(face_image)
            # plt.show()
            return face_image
        else:
            return image
    

class Trainer:
    def __init__(self, save_path):
        """
        Initialize the Trainer with data transformations, datasets, model, and optimization components.
        
        Args:
        save_path (str): Directory to save model checkpoints.
        """
        # Create an instance of the YOLO face detection model
        face_model = YOLO('yolov8n-face.pt', verbose=False)

        # Create an instance of the CropFaceTransform class
        crop_face_transform = CropFaceTransform(face_model)

        # Define image transformations
        self.transform = transforms.Compose([
            crop_face_transform,
            transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
            transforms.RandomRotation(90),               # Apply random rotation
            transforms.RandomAdjustSharpness(2),         # Adjust sharpness randomly
            transforms.RandomHorizontalFlip(0.5),        # Random horizontal flip
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
        self.model = timm.create_model('timm/convnextv2_pico.fcmae_ft_in1k', pretrained=True)
        #print(self.model)
        self.model.head.fc = nn.Linear(512, 6)  # Modify the final layer for 8 classes
        
        # Define loss function
        # CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one single class
        self.criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        # Adam optimizer with a learning rate of 0.00001
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)
        
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
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(self.train_loader)}")
            
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

# Usage
save_path = "checkpoints/convnext"
trainer = Trainer(save_path)
trainer.train(1000)