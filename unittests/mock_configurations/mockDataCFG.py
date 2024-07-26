class MockDataCFG:
    """Class that holds information about the projects data"""
    time_column: str = 'date'
    target_column = 'orders'

    usable_columns: list[str] = ['warehouse', 'date']

    ordinal_columns: list[str] = ['day_of_week', 'year']
    categorical_columns: list[str] = ['month_name', 'warehouse']

    target_encoding_cols: list[str] = []
    standardize_columns: list[str] = []
