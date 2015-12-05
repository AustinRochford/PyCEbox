import numpy as np
import pandas as pd


def get_grid_points(x, num_grid_points):
    if num_grid_points is None:
        return x.unique()
    else:
        # unique is necessary, because if num_grid_points is too much larger
        # than x.shape[0], there will be duplicate quantiles (even with
        # interpolation)
        return x.quantile(np.linspace(0, 1, num_grid_points)).unique()


def ice(data, column, predict, num_grid_points=None):
    """
    Generate ICE curves for given column and model.

    predict is a function generating the model predictions that accepts a DataFrame with the same number of columns as
    data.

    If num_grid_points is None, column varies over its unique values in data.
    If num_grid_points is not None, column varies over num_grid_pts values, evenly spaced at its quantiles in data.
    """
    x_s = get_grid_points(data[column], num_grid_points)
    ice_data = to_ice_data(data, column, x_s)
    ice_data['ice_y'] = predict(ice_data)

    other_columns = list(data.columns)
    other_columns.remove(column)
    ice_df = ice_data.pivot_table(values='ice_y', index=other_columns, columns=column).T

    return ice_df


def to_ice_data(data, column, x_s):
    """
    Create the DataFrame necessary for ICE calculations
    """
    ice_data = pd.DataFrame(np.repeat(data.values, x_s.size, axis=0), columns=data.columns)
    ice_data[column] = np.tile(x_s, data.shape[0])

    return ice_data
