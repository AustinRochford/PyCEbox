import nose
import numpy as np
import pandas as pd

from pycebox import ice


def test_to_ice_data():
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    data = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    x_s = np.array([10, 11])

    ice_data = ice.to_ice_data(data, 'x3', x_s)
    ice_data_expected = pd.DataFrame(np.array([[1, 2, 10],
                                               [1, 2, 11],
                                               [4, 5, 10],
                                               [4, 5, 11],
                                               [7, 8, 10],
                                               [7, 8, 11]]),
                                     columns=['x1', 'x2', 'x3'])

    assert (ice_data == ice_data_expected).all().all()


def test_to_ice_data_one_sample():
    X = np.array([[1, 0]])
    data = pd.DataFrame(X, columns=['x1', 'x2'])
    x_s = np.arange(5)

    ice_data = ice.to_ice_data(data, 'x2', x_s)
    ice_data_expected = pd.DataFrame({'x1': np.ones(5), 'x2': np.arange(5)})

    assert (ice_data == ice_data_expected).all().all()


def test_to_ice_data_one_test_point():
    X = np.eye(3)
    data = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    x_s = np.array([-1])

    ice_data = ice.to_ice_data(data, 'x1', x_s)
    ice_data_expected = pd.DataFrame({
        'x1': np.array([-1, -1, -1]),
        'x2': np.array([0, 1, 0]),
        'x3': np.array([0, 0, 1])
    })

    assert (ice_data == ice_data_expected).all().all()
