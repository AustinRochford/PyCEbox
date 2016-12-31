from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from itertools import repeat
from six.moves.builtins import range

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use('AGG') # make the tests pass flexible on OS X
from pycebox import ice


# helpers
def compare_with_NaN(x, y):
    return (x == y) | (np.isnan(x) & np.isnan(y))


# tests
def test_get_grid_points_num_grid_points():
    x = pd.Series(np.array([0, 0, 1, 2, 3, 4, 5, 6, 7]))
    expected_grid_points = np.array([0, 1, 3, 5, 7])

    assert (expected_grid_points == ice.get_grid_points(x, 5)).all()


def test_get_grid_points_num_grid_points_too_many():
    x = pd.Series(np.array([0, 0, 1]))
    expected_grid_points = np.array([0, 0.5, 1])

    assert (expected_grid_points == ice.get_grid_points(x, 5)).all()


@given(st.lists(st.floats()))
def test_get_grid_points_num_grid_points_None(l):
    x = pd.Series(l)

    assert compare_with_NaN(x.unique(), ice.get_grid_points(x, None)).all()


def test_ice_one_sample_one_point():
    X = np.array([[0, 1]])
    df = pd.DataFrame(X, columns=['x0', 'x1'])

    ice_df = ice.ice(df, 'x1', lambda X: -1)
    expected_columns = pd.MultiIndex.from_tuples([(1, 0)], names=['data_x1', 'x0'])
    ice_df_expected = pd.DataFrame(np.array([-1]),
                                   columns=expected_columns,
                                   index=pd.Series(1, name='x1'))

    assert (ice_df == ice_df_expected).all().all()


def test_ice_two_samples_two_points():
    X = np.eye(2)
    df = pd.DataFrame(X, columns=['x0', 'x1'])

    ice_df = ice.ice(df, 'x0', lambda X: X.prod(axis=1))

    expected_columns = pd.MultiIndex.from_tuples([(0, 1), (1, 0)], names=['data_x0', 'x1'])
    ice_df_expected = pd.DataFrame(np.array([[0, 0],
                                             [1, 0]]),
                                   columns=expected_columns,
                                   index=pd.Series([0., 1.], name='x0'))

    assert (ice_df == ice_df_expected).all().all()


def test_ice_num_grid_points():
    X = np.eye(3)
    df = pd.DataFrame(X, columns=['x0', 'x1', 'x2'])
    
    ice_df = ice.ice(df, 'x2', lambda X: X.dot(np.array([[1., 2., 3.]]).T),
                     num_grid_points=5)

    expected_columns = pd.MultiIndex.from_tuples([(0, 0, 1), (0, 1, 0), (1, 0, 0)], names=['data_x2', 'x0', 'x1'])
    ice_df_expected = pd.DataFrame(np.array([[2., 1., 0.],
                                             [3.5, 2.5, 1.5],
                                             [5., 4., 3.]]),
                                   columns=expected_columns,
                                   index=pd.Series([0., 0.5, 1.], name='x2'))

    assert (ice_df == ice_df_expected).all().all()


@given((st.tuples(st.integers(min_value=2, max_value=10),
                  st.integers(min_value=2, max_value=10))
          .flatmap(lambda shape: st.tuples(arrays(np.float64, shape[0]),
                                           arrays(np.float64, shape)))))
def test_pdp(args):
    index, l = args
    index.sort()
    ice_df = pd.DataFrame(l, index=index)

    pdp = ice.pdp(ice_df)
    pdp_expected = pd.Series([pd.Series(row).mean() for row in l], index=index)

    assert compare_with_NaN(pdp, pdp_expected).all()


def test_to_ice_data():
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    data = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
    x_s = np.array([10, 11])

    ice_data, orig_column = ice.to_ice_data(data, 'x3', x_s)
    ice_data_expected = pd.DataFrame(np.array([[1, 2, 10],
                                               [1, 2, 11],
                                               [4, 5, 10],
                                               [4, 5, 11],
                                               [7, 8, 10],
                                               [7, 8, 11]]),
                                     columns=['x1', 'x2', 'x3'])

    orig_column_expected = pd.Series([3, 3, 6, 6, 9, 9], name='x3')

    assert (ice_data == ice_data_expected).all().all()
    assert compare_with_NaN(orig_column, orig_column_expected).all()


@given((st.tuples(st.integers(min_value=2, max_value=10))
          .flatmap(lambda size: arrays(np.float64, (1,) + size))),
       (st.tuples(st.integers(min_value=1, max_value=10))
          .flatmap(lambda shape: arrays(np.float64, shape))))
def test_to_ice_data_one_sample(X, x_s):
    n_cols = X.shape[1]
    columns = ['x{}'.format(i) for i in range(n_cols)]
    data = pd.DataFrame(X, columns=list(columns))

    ice_data, orig_column = ice.to_ice_data(data, 'x1', x_s)

    ice_data_expected_values = np.repeat(X, x_s.size, axis=0)
    ice_data_expected_values[:, 1] = x_s
    ice_data_expected = pd.DataFrame(ice_data_expected_values, columns=columns)

    orig_column_expected = np.repeat(data['x1'].values, x_s.size)

    assert compare_with_NaN(ice_data, ice_data_expected).all().all()
    assert compare_with_NaN(orig_column, orig_column_expected).all()


# generate a list of length m of lists of length n of floats, to turn into a 2d numpy array
@given((st.tuples(st.integers(min_value=2, max_value=10),
                  st.integers(min_value=2, max_value=10))
          .flatmap(lambda shape: arrays(np.float64, shape))),
       st.floats())
def test_to_ice_data_one_test_point(l, x_s):
    X = np.array(l)
    n_cols = X.shape[1]
    columns = ['x{}'.format(i) for i in range(n_cols)]
    data = pd.DataFrame(X, columns=columns)
    x_s = np.array(x_s)

    ice_data, orig_column = ice.to_ice_data(data, 'x0', x_s)

    ice_data_expected_values = X.copy()
    ice_data_expected_values[:, 0] = x_s
    ice_data_expected = pd.DataFrame(ice_data_expected_values, columns=columns)

    orig_column_expected = data['x0']

    assert compare_with_NaN(ice_data, ice_data_expected).all().all()
    assert compare_with_NaN(orig_column, orig_column_expected).all()
