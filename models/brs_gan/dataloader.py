from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class DataLoader:
    def __init__(self):
        self.batch_size = None

    def next_batch(self):
        pass

class MNISTDataLoader(DataLoader):
    def __init__(self, batch_size, mode="smooth"):
        super().__init__()
        self.batch_size = batch_size
        self.mode = mode
        self.mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)

    def next_batch(self):
        if self.mode == "smooth" or self.mode == "gradient":
            X_t, label = self.mnist.train.next_batch(self.batch_size)
        elif self.mode == "binary":
            X_t, label = self.mnist.train.next_batch(self.batch_size)
            X_t = (X_t > 0.5).astype(np.float32)
        elif self.mode == "multilevel":
            X_mb_s, label = self.mnist.train.next_batch(self.batch_size)
            X_t = np.zeros((self.batch_size, 784))
            for j in range(1, 10):
                X_t = X_t + (X_mb_s > j / 10.0).astype(float)
            X_t = X_t / 10.0
        else:
            print("Incompatiable mode!")
            exit()
        return X_t, label

class SyntheticDataLoader(DataLoader):
    def __init__(self, batch_size, mode="smooth"):
        super().__init__()
        self.batch_size = batch_size
        self.mu = np.random.uniform(-50, 50, [10, 10])
        self.sigma = np.random.uniform(1, 10, [10])

    def next_batch(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        X_t = []
        c = np.random.randint(10, size=batch_size)
        for i in range(batch_size):
            X_t.append(np.random.normal(loc = self.mu[c[i]], scale = self.sigma[c[i]]))
        return X_t, c
