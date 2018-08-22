import numpy as np
import pandas as pd

dataset = pd.read_csv('./data/fer2013.csv')

train_images = []
train_labels = []
val_images = []
val_labels = []
test_images = []
test_labels = []


def to_one_hot(labels, classes):
    N = len(labels)
    temp = np.zeros((N, classes))
    temp[np.arange(N), labels] = 1
    return np.array(temp, dtype=np.int32)


for data in dataset.values:
    image_data = np.array(data[1].split(), dtype=np.float32)
    image_data = image_data.reshape(48, 48, 1)
    label = int(data[0])

    if data[2] == 'Training':
        train_images.append(image_data)
        train_labels.append(label)
    elif data[2] == 'PublicTest':
        val_images.append(image_data)
        val_labels.append(label)
    elif data[2] == 'PrivateTest':
        test_images.append(image_data)
        test_labels.append(label)


train_images = np.array(train_images, dtype=np.float32)
val_images = np.array(val_images, dtype=np.float32)
test_images = np.array(test_images, dtype=np.float32)

train_labels = to_one_hot(train_labels, 7)
val_labels = to_one_hot(val_labels, 7)
test_labels = to_one_hot(test_labels, 7)

print(train_images.shape)
print(train_labels.shape)
print(val_images.shape)
print(val_labels.shape)
print(test_images.shape)
print(test_labels.shape)

np.savez_compressed('./data/train.npz', images=train_images, labels=train_labels)
np.savez_compressed('./data/validation.npz', images=val_images, labels=val_labels)
np.savez_compressed('./data/test.npz', images=test_images, labels=test_labels)
