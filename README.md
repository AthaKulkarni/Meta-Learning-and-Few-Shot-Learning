# Meta Learning and Few Shot Learning

## Individual Project 4

**Atharva Pradip Kulkarni**  
**Masters in Data Science**  
**Worcester Polytechnic Institute**  
**DS 504 / CS 586: Big Data Analytics**  
**Professor Dr. Yanhua Li**  
**April 23, 2024**

---

## 1. Introduction and Proposal

Siamese neural networks are a specialized neural network architecture designed for comparing two inputs to measure their similarity. They are commonly used in tasks such as image or speech recognition, where determining whether two inputs are the same or similar is crucial. These networks consist of two identical sub-networks that share weights, allowing simultaneous processing of both inputs and producing similar feature representations. A significant advantage of Siamese networks is their ability to perform few-shot learning, enabling the network to learn new tasks with minimal examples, making them particularly useful in scenarios with limited labeled data.

This project focuses on the implementation of a Siamese neural network for similarity detection tasks, particularly dealing with complex geospatial trajectory data. The dataset consists of 500 key-value pairs stored in a dictionary, where each key represents a driver or their plate number linked to a list of trajectories. These lists contain nested data structured in three dimensions, making the dataset resource-intensive, with a requirement of more than 28 GB of RAM. Understanding and processing this intricate data posed significant challenges, adding to the complexity of the project.

## 2. Methodology

### 2.1 Feature Generation

The geospatial trajectory data was preprocessed and saved in a pickle file to make it more usable for future analysis or modeling. The script imports essential libraries for data manipulation and file handling, and various utility functions are provided to assist in tasks like time code assignment, area block estimation, distance calculation, and feature engineering. The trajectories are processed and divided into subsequences of a specific length, with error handling to manage any issues during processing. The processed trajectories are then saved in a new pickle file for future use.

### 2.2 Network Structure

The SiameseLSTM neural network model is designed for training Siamese Networks, which are effective in tasks where comparing the similarity between input pairs is essential. The model utilizes LSTM layers to capture temporal dependencies within sequential data. After LSTM processing, fully connected layers refine the extracted features, and the model computes the L1 distance between feature representations of the two trajectories. A final fully connected layer with a sigmoid activation function produces a probability score representing the similarity between the input trajectories.

### 2.3 Training and Validation Process

The training process was implemented using PyTorch. The training and validation datasets were split and processed using DataLoader objects. The SiameseLSTM model was trained using the Adam optimizer and Binary Cross-Entropy Loss. The training process involved iterating over the datasets, computing loss and accuracy, and updating model parameters. The model's state was saved at the epoch with the highest validation accuracy. The training framework is robust and scalable, facilitating tasks such as similarity comparison and pattern recognition.

## 3. Evaluation and Results

### 3.1 Training Results

The Siamese neural network was trained on a dataset to evaluate its ability to detect similarities between pairs of trajectories. Various hyperparameters, including learning rate, epochs, and batch size, were experimented with to optimize accuracy. The highest accuracy of 88.92% was achieved with a learning rate of 0.001, 90 epochs, and a batch size of 32. Below are the results of different experiments:

| Exp No. | Epochs | Learning Rate | Batch Size | Accuracy (%) |
|---------|--------|---------------|------------|--------------|
| 1       | 5      | 0.01          | 32         | 64.58        |
| 2       | 10     | 0.01          | 32         | 75.8         |
| 3       | 20     | 0.01          | 32         | 77.08        |
| 4       | 30     | 0.01          | 32         | 80.4         |
| 5       | 40     | 0.01          | 32         | 69.65        |
| 6       | 40     | 0.001         | 32         | 76.48        |
| 7       | 50     | 0.001         | 32         | 81.6         |
| 8       | 60     | 0.001         | 32         | 84.7         |
| 9       | 70     | 0.001         | 32         | 86.25        |
| 10      | 90     | 0.001         | 32         | 88.92        |

### 3.2 Testing Model

The trained model was evaluated on a test dataset to assess its performance on unseen data. The evaluation involved computing the test loss and accuracy by comparing the model's predictions with the ground truth labels. The model achieved high accuracy in similarity detection, demonstrating its effectiveness in handling complex geospatial trajectory data.

## 4. Conclusion

This project explored the use of Siamese neural networks for similarity detection tasks, particularly in the context of geospatial trajectory data. The project successfully implemented a SiameseLSTM model and optimized its performance through careful experimentation with hyperparameters. The project highlights the importance of meta-learning and few-shot learning in scenarios with limited labeled data, demonstrating the versatility and robustness of Siamese networks in such environments. The methodologies and insights gained from this project can be applied to various domains requiring similarity detection and pattern recognition.
