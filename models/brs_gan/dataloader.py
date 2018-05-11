from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class DataLoader:
    def __init__(self):
        self.batch_size = None

    def next_batch(self):
        pass

class MNISTDataLoader(DataLoader):
    def __init__(self, batch_size, mode="smooth"):
        super.init()
        self.batch_size = batch_size
        self.mode = mode
        self.mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)

    def next_batch(self):
        if self.mode == "smooth" or self.mode == "gradient":
            X_t, _ = self.mnist.train.next_batch(self.batch_size)
        elif self.mode == "binary":
            X_t, _ = self.mnist.train.next_batch(self.batch_size)
            X_t = (X_t > 0.5).astype(np.float32)
        elif self.mode == "multilevel":
            X_mb_s, _ = self.mnist.train.next_batch(self.batch_size)
            X_t = np.zeros((self.batch_size, 784))
            for j in range(1, 10):
                X_t = X_t + (X_mb_s > j / 10.0).astype(float)
            X_t = X_t / 10.0
        else:
            print("Incompatiable mode!")
            exit()

class SyntheticDataLoader(DataLoader):
    def __init__(self, batch_size, mode="smooth"):
        super.init()
        self.batch_size = batch_size
        

    def next_batch(self):
        if self.mode == "smooth" or self.mode == "gradient":
            X_t, _ = self.mnist.train.next_batch(self.batch_size)
        elif self.mode == "binary":
            X_t, _ = self.mnist.train.next_batch(self.batch_size)
            X_t = (X_t > 0.5).astype(np.float32)
        elif self.mode == "multilevel":
            X_mb_s, _ = self.mnist.train.next_batch(self.batch_size)
            X_t = np.zeros((self.batch_size, 784))
            for j in range(1, 10):
                X_t = X_t + (X_mb_s > j / 10.0).astype(float)
            X_t = X_t / 10.0
        else:
            print("Incompatiable mode!")
            exit()