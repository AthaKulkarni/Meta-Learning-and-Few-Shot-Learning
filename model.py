import torch
import torch.nn as nn

class SiameseLSTM(nn.Module):
    """
    Neural network model for training Siamese Networks.
    """
    def __init__(self, input_dim, hidden_dim):
        super(SiameseLSTM, self).__init__()

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Fully connected layers for comparison
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # Split the input tensor along the second dimension
        x1, x2 = x[:, 0, :, :], x[:, 1, :, :]

        # Pass both trajectories through LSTM
        out1, _ = self.lstm(x1)
        out2, _ = self.lstm(x2)

        # Apply fully connected layers for comparison
        out1 = torch.relu(self.fc1(out1[:, -1, :]))  # Get the last output of LSTM
        out1 = torch.relu(self.fc2(out1))
        
        out2 = torch.relu(self.fc1(out2[:, -1, :]))  # Get the last output of LSTM
        out2 = torch.relu(self.fc2(out2))

        # Calculate L1 distance
        distance = torch.abs(out1 - out2)

        # Apply final fully connected layer
        out = self.fc3(distance)
        
        # Apply sigmoid activation to squash the output between 0 and 1
        out = self.sigmoid(out)

        return out
