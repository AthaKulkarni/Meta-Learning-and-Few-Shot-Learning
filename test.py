# test_model.py

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from model import SiameseLSTM
from tqdm import tqdm

from train import load_data, train_model, evaluate, SiameseDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 6
input_dim = 6  # Number of features
hidden_dim = 64  # Hidden dimension of LSTM
criterion = nn.BCELoss()

val_pairs, val_labels = load_data(filename='X_Y_validation20_pairs.pkl')
test_dataset = SiameseDataset(val_pairs, val_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = SiameseLSTM(input_dim, hidden_dim).to(device)
model.load_state_dict(torch.load('best_acc_model.pt'))
model.to(device)

def test_model(model, test_loader, criterion):
    """
    Function to test the model performance on the test set.
    Computes loss and accuracy on the test set.
    
    Parameters:
        model: SiameseLSTM model
        test_loader: DataLoader for test data
        criterion: Loss function (e.g., BCELoss)
        
    Returns:
        test_loss: Test loss
        test_accuracy: Test accuracy
    """
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))  # BCELoss expects 1D target
            test_loss += loss.item()
            
            # Compute test accuracy
            predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
            correct += (predicted == targets.unsqueeze(1)).sum().item()
            total += targets.size(0)
    
    test_accuracy = correct / total
    average_test_loss = test_loss / len(test_loader)
    
    print(f"Test Loss: {average_test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    
    return average_test_loss, test_accuracy
    ###########################

test_loss, test_acc = test_model(model, test_loader,  criterion)