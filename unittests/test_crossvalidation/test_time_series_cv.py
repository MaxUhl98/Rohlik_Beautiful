import pytest
from cross_validation.time_series_cv import *
import pandas as pd


def test_get_sorted_timeseries_array():
    date_series = pd.Series(['2024-01-01', '2023-01-01', '2024-01-02', '2023-01-02', '2024-02-03', '2024-02-05'])
    to_test = get_sorted_timeseries_array(date_series)
    sorted_array = np.array(['2023-01-01', '2023-01-02', '2024-01-01', '2024-01-02', '2024-02-03', '2024-02-05'])
    assert (to_test == sorted_array).all()

