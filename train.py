import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import SiameseLSTM 
import matplotlib.pyplot as plt # Assuming your model is defined in a file named model.py
import numpy as np
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the SiameseDataset class
class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs  # pairs is a numpy array of shape (n_samples, 2, seq_len, features)
        self.labels = labels  # labels is a numpy array of shape (n_samples,)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.pairs[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    
def load_data(filename='X_Y_train400_pairs.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    pairs = data['pairs']
    labels = data['labels']
    return pairs, labels

batch_size = 32
input_dim = 6  # Number of features
hidden_dim = 64  # Hidden dimension of LSTM
num_epochs = 90
learning_rate = 0.001

pairs, labels = load_data(filename='X_Y_train400_pairs.pkl')


train_pairs, test_pairs, train_labels, test_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)

train_dataset = SiameseDataset(pairs, labels)
val_dataset = SiameseDataset(test_pairs, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = SiameseLSTM(input_dim, hidden_dim).to(device)

model = SiameseLSTM(input_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

def train(model, optimizer, criterion, train_loader):
    """
    Function to handle the training of the model.
    Iterates over the training dataset and updates model parameters.
    
    Parameters:
        model: SiameseLSTM model
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (e.g., BCELoss)
        train_loader: DataLoader for training data
        
    Returns:
        train_loss: Average loss over the training dataset
        train_acc: Training accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))  # BCELoss expects 1D target
        loss.backward()
        optimizer.step()
        
        # Compute training accuracy
        predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
        correct += (predicted == targets.unsqueeze(1)).sum().item()
        total += targets.size(0)
        
        total_loss += loss.item()

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)
    
    return train_loss, train_acc

def evaluate(model, criterion, loader):
    """
    Function to evaluate the model performance on the validation set.
    Computes loss and accuracy without updating model parameters.
    
    Parameters:
        model: SiameseLSTM model
        criterion: Loss function (e.g., BCELoss)
        loader: DataLoader for validation data
        
    Returns:
        test_loss: Loss on validation set
        test_accuracy: Accuracy on validation set
    """
    device = next(model.parameters()).device  # Get device of model's parameters
    
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move inputs to device
            
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets.unsqueeze(1))  # Calculate loss
            
            # Calculate validation accuracy
            predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
            correct += (predicted == targets.unsqueeze(1)).sum().item()
            total += targets.size(0)
            
            test_loss += loss.item()
    
    # Calculate average loss and accuracy
    average_loss = test_loss / len(loader)
    accuracy = correct / total
    
    return average_loss, accuracy

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs):
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    
    Parameters:
        model: SiameseLSTM model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer (e.g., Adam)
        criterion: Loss function (e.g., BCELoss)
        num_epochs: Number of training epochs
        
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to appropriate device
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move inputs to device
            
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets.unsqueeze(1))  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            # Calculate training accuracy
            predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
            correct_train += (predicted == targets.unsqueeze(1)).sum().item()
            total_train += targets.size(0)
            
            train_loss += loss.item()
        
        # Calculate average training loss and accuracy for the epoch
        average_train_loss = train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        
        # Evaluate on validation data
        val_loss, val_accuracy = evaluate(model, criterion, val_loader)
        
        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Training Loss: {average_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print("=" * 40)
        
        # Save losses and accuracies for plotting
        train_losses.append(average_train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    torch.save(model.state_dict(), 'best_acc_model.pt')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Function to plot the graphs of training and validation accuracies and losses.
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot training and validation losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs)
    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies)
