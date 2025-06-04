import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
import time
import pandas as pd
import numpy as np
import torch
import socket
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import struct
import pandas as pd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv(f'/home/kdd1/shared_folder/UNSW/UNSW_results_stephen/final_UNSW_pequals1.csv')

start_preprocess_time = time.time()

string_cols = [col for col in df.columns if df[col].dtype == object]
le = LabelEncoder()
df[string_cols] = df[string_cols].apply(lambda col: le.fit_transform(col))

# Separate the features and labels
features = df.iloc[:, :-1].values
features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
labels = df.iloc[:, -1].values

# Normalize the features using StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=4)

# Print the size of each set
print("Training features shape: ", train_features.shape)
print("Training labels shape: ", train_labels.shape)
print("Testing features shape: ", test_features.shape)
print("Testing labels shape: ", test_labels.shape)

import numpy as np

# Count the number of instances for each label in the training set
unique_labels_train, counts_train = np.unique(train_labels, return_counts=True)
for label, count in zip(unique_labels_train, counts_train):
    print(f"Training set: Label {label} has {count} instances")

# Count the number of instances for each label in the testing set
unique_labels_test, counts_test = np.unique(test_labels, return_counts=True)
for label, count in zip(unique_labels_test, counts_test):
    print(f"Testing set: Label {label} has {count} instances")


# Convert the training and testing data into PyTorch Tensors
train_features_tensor = torch.from_numpy(train_features).float()
train_labels_tensor = torch.from_numpy(train_labels).long()
test_features_tensor = torch.from_numpy(test_features).float()
test_labels_tensor = torch.from_numpy(test_labels).long()

# Create a PyTorch TensorDataset for the training data
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)

test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)

end_preprocess_time = time.time()

total_preprocessing_time = end_preprocess_time - start_preprocess_time
print(f"Elapsed preprocessing time: {total_preprocessing_time} seconds")

# Create a PyTorch DataLoader for the training data
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1,3), padding=(0,1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1,3), padding=(0,1))
        self.fc1 = nn.Linear(64 * 1504, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 1504, 1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 64 * 1504)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


# Create an instance of the model and pass a batch of inputs
model = Net()
inputs, targets = next(iter(train_dataloader))

outputs = model(inputs)
# find the unique labels in the target tensor
unique_labels = torch.unique(train_labels_tensor)

# print the unique labels
print(unique_labels)
# Print the shape of the output tensor
print(outputs.shape)


start_training_testing_time = time.time()

model = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model for 10 epochs
for epoch in range(30):
    for i, (inputs, labels) in enumerate(train_dataloader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Print the loss every 100 batches
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, 30, i+1, len(train_dataloader), loss.item()))

torch.save(model.state_dict(), f'unsw_1.pth')


  # validate the model
correct = 0
total = 0
with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            images, labels = images, labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
print("Accuracy of the network on the test data: {:.2f}%".format(100 * correct / total))

print("Finished training")

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
# Make predictions on the test set
y_true = []
y_pred = []
with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(y.numpy())
        y_pred.extend(predicted.numpy())

# Compute precision, recall, and F1 score
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

print('Precision:', precision*100)
print('Recall:', recall*100)
print('F1 score:', f1*100)

label_map = {0: 'analysis', 1: 'backdoor', 2: 'dos', 3: 'exploits', 4: 'fuzzers', 5: 'generic', 6: 'normal', 7:'reconnaissance', 8: 'shellcode', 9: 'worms'}



# Convert y_true and y_pred to lists
y_true_list = list(y_true)
y_pred_list = list(y_pred)

# Map the label values to their corresponding names using list comprehension
y_true_mapped = [label_map[y] for y in y_true_list]
y_pred_mapped = [label_map[y] for y in y_pred_list]

# Generate the classification report
print(classification_report(y_true_mapped, y_pred_mapped, zero_division=1))
print('\n')
print('                    Confusion Matrix')




end_training_testing_time = time.time()

total_training_testing_time = end_training_testing_time - start_training_testing_time

print(f"Elapsed training and testing time: {total_training_testing_time:.2f} seconds.")