# Import modules
import numpy as np
import pytest


@pytest.fixture()
def particles():
    shape = (np.random.randint(10, 20), np.random.randint(2, 6))
    particles_ = np.random.uniform(0, 10, shape)
    print(particles_)
    return particles_
