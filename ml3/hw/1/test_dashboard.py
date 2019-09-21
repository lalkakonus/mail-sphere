import pytest
import numpy as np
from dashboard import DashBoard

@pytest.fixture
def data_1():
    x = np.linspace(0, 300, 300)
    y = 100 / (x ** 0.5 + 1)
    y += np.random.random(300)
    return x, y 

@pytest.fixture
def data_2():
    x = np.linspace(0, 300, 300)
    y = -7 + 70 / np.log(1 + x)
    y += np.random.random(300)
    y += np.sin(x)
    return x, y 

def test_regressor(data_1, data_2):
    board = DashBoard()
    x, y = data_1
    board.init_graph(name="baseline", title="sklearn", line_type="baseline", c="g")
    for i in range(x.size):
        board.add_point("baseline", y[i], x[i])

    x, y = data_2
    board.init_graph(name="another", title="my one", x=x, y=y, c="r")
    
    board.make_plot()
