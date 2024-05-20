# MNIST Handwritten Digit Recognition

- Dataset Source: [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/index.html) or [MNIST Dataset from Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

## Using **tensorflow** and **keras**

- Note: We can use tensorflow to load the dataset so no need for downloading :blush:.

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
```

- More info: [tf.keras.datasets.mnist.load_data](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data)
- We will use a Sequential model with 3 Dense layers for this task :thumbsup:.

## Using **torch**

- We can download the dataset using torch.

```python
import torch
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

# Transform
transform = transforms.Compose([
    transforms.Resize([28, 28]),  # [Height, Width]
    transforms.ToTensor()  # PIL image to torch.tensor
])

# Dataset
training_data = MNIST(root="./data", train=True, transform=transform, target_transform=None, download=True)
test_data = MNIST(root="./data", train=False, transform=transform, target_transform=None, download=True)
```

- torch offers great functionalities and versatility but seems tedious :thumbsup:|:thumbsdown:.