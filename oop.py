from abc import ABC, abstractmethod
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, input_data):
        pass

class CNNClassifier(DigitClassificationInterface):
    def __init__(self):
        # Define a simple CNN architecture for demonstration
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.fc1 = nn.Linear(64 * 28 * 28, 10)
            
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                return x
        
        self.model = SimpleCNN()

        
    def predict(self, input_data):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output = self.model(input_tensor)
            predicted_label = torch.argmax(output, dim=1).item()
        return predicted_label


class RFClassifier(DigitClassificationInterface):
    def __init__(self):
        # Random Forest model (Assume it is already trained)
        self.model = RandomForestClassifier(n_estimators=100)
        
    def predict(self, input_data):
        input_flattened = input_data.flatten().reshape(1, -1)
        return self.model.predict(input_flattened)[0]


class RandomClassifier(DigitClassificationInterface):
    def predict(self, input_data):
        # Crop the 10x10 center from the 28x28 input
        center_crop = input_data[9:19, 9:19]

        return random.randint(0, 9)


class DigitClassifier:
    def __init__(self, algorithm):
        if algorithm == 'cnn':
            self.classifier = CNNClassifier()
        elif algorithm == 'rf':
            self.classifier = RFClassifier()
        elif algorithm == 'rand':
            self.classifier = RandomClassifier()
        else:
            raise ValueError("Unsupported algorithm. Choose from 'cnn', 'rf', or 'rand'.")
    
    def predict(self, input_data):
        if input_data.shape != (28, 28):
            raise ValueError("Input data must be a 28x28 array.")
        return self.classifier.predict(input_data)

    def train(self, X, y):
      raise NotImplementedError()