import numpy as np
import pandas as pd

def to_ice_data(data, column, x_s):
    """
    Create the DataFrame necessary for ICE calculations
    """
    ice_data = pd.DataFrame(np.repeat(data.values, x_s.size, axis=0), columns=data.columns)
    ice_data[column] = np.tile(x_s, data.shape[0])

    return ice_data
