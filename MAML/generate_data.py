import numpy as np
import math
import matplotlib.pyplot as plt

class DataGenerator(object):
    def __init__(self, batch):
        self.batch = batch
        self.amp = np.random.uniform(0.1, 5, batch)
        self.phase = np.random.uniform(0, np.pi, batch)
        self.batch_pointer = -1
    
    def sample(self, n):
        fun = np.vectorize(
            lambda x: 
            self.amp[self.batch_pointer]*math.sin(x+self.phase[self.batch_pointer]))
        x = np.random.uniform(-5, 5, n)
        return x.reshape(-1, 1), fun(x).reshape(-1, 1)

    def __call__(self):
        for i in range(self.batch):
            self.batch_pointer += 1
            yield self.batch_pointer
        

if __name__ == "__main__":
    # test `DataGenerator`
    dg = DataGenerator(batch=3)
    for i in dg():
        x, y = dg.sample(10)
        plt.scatter(x, y)
    plt.show()
