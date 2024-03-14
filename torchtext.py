import torch
from torchtext import datasets as data
import random

# Download and load the AG_NEWS dataset
train_dataset, test_dataset = data.AG_NEWS(root='.', split=('train', 'test'))

# Get a random example from the training dataset
random_index = random.randint(0, len(train_dataset) - 1)
label, text = train_dataset[random_index]

# Print some information about the random example
print(f"Label: {label}")
print(f"Text: {text}")
