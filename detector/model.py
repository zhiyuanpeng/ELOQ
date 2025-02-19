import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes=1, dropout=0.1):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_size, 128) 
        self.dropout1 = nn.Dropout(dropout)  
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x: (B, E)
        x = F.relu(self.fc1(x))  # Apply activation function
        x = self.dropout1(x)  # Apply dropout
        x = torch.sigmoid(self.fc2(x))  # Binary classification
        return x
    