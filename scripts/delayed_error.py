import ray
import numpy as np
from sklearn.linear_model import ElasticNet


model = ElasticNet()
melih_size = 10
X, y = np.random.randn(melih_size, 2), np.random.randint(2, size=melih_size)
X[1, 1] = np.nan

@ray.remote
def foo():
    model.fit(X, y)

@ray.remote
def bar():
    return model.predict(X[0:2])

foo.remote()
print("No errors yet...")
ray.get(bar.remote()) # Error only raised on last task when ray.get() called
