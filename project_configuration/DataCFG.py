from project_configuration.ConfigurationFunctionalities import BasicFunctionalities


class DataCFG(BasicFunctionalities):
    """Class that holds information about the projects data"""
    train_path: str = 'data/train.csv'
    test_path: str = 'data/test.csv'

    time_column: str = 'date'
    target_column: str = 'orders'
    group_column: str = 'warehouse'

    usable_columns: list[str] = ['warehouse', 'date', 'holiday_name', 'holiday', 'shops_closed',
                                 'winter_school_holidays', 'school_holidays']

    ordinal_columns: list[str] = ['day_of_week', 'year', 'week', 'day']
    categorical_columns: list[str] = ['month', 'warehouse', 'holiday_name']

    target_encoding_cols: list[str] = []
    standardize_columns: list[str] = []

    def __init__(self):
        self.train_path = self.train_path
        self.test_path = self.test_path
        self.time_column = self.time_column
        self.target_column = self.target_column
        self.group_column = self.group_column
        self.usable_columns = self.usable_columns
        self.ordinal_columns = self.ordinal_columns
        self.categorical_columns = self.categorical_columns
        self.target_encoding_cols = self.target_encoding_cols
        self.standardize_columns = self.standardize_columns
