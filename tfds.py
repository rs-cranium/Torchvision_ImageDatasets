import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
dataset, info = tfds.load('cifar10', split='train', with_info=True)

# Function to normalize and resize images
def preprocess(example):
    image = tf.image.resize(example['image'], (32, 32))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, example['label']

# Apply preprocessing and shuffle the dataset
dataset = dataset.map(preprocess).shuffle(1000)

# Visualize some sample images
plt.figure(figsize=(10, 10))
for i, example in enumerate(dataset.take(9)):
    image, label = example
    plt.subplot(3, 3, i+1)
    plt.imshow(image)
    plt.title('Label: {}'.format(info.features['label'].int2str(label.numpy())))
    plt.axis('off')
plt.show()
