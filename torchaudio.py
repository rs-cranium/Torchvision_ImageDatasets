import torch
import torchaudio
from torchaudio.datasets import YESNO
import random

# Download and load the YESNO dataset
dataset = YESNO('.', download=True)

# Get a random example from the dataset
random_index = random.randint(0, len(dataset) - 1)
waveform, sample_rate, label = dataset[random_index]

# Print some information about the random example
print(f"Sample rate: {sample_rate}")
print(f"Label: {label}")
print(f"Waveform shape: {waveform.size()}")
