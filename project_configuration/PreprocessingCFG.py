from preprocessing.feature_engineering import *
from typing import *


class PreprocessingCFG:
    # Encoding settings
    use_onehot: bool = True
    use_ordinal_encoding: bool = True

    # Preprocessing Settings
    use_basic_timeseries_preprocessing: bool = True
    standardize: bool = False

    # OpenFE settings
    use_openfe: bool = True
    openfe_kwargs: dict[str, Any] = {'feature_boosting': True,
                                     'task_type': 'regression',
                                     'n_repeats': 2,
                                     'n_jobs': 4}
    openfe_feature_save_directory: str = 'preprocessing/openfe_features'

    # Feature Engineering Settings
    specialized_feature_engineering_function: Union[
        Callable, None] = engineer_rohlik_specific_features  # Set to None if you don't want specialized preprocessing

    data_save_directory: str = 'preprocessing/preprocessed_datasets'

    def __init__(self):
        """Initialization of class attributes which adds 'self' elements to self.__dict__"""
        self.use_onehot = self.use_onehot
        self.use_ordinal_encoding = self.use_ordinal_encoding
        self.use_basic_timeseries_preprocessing = self.use_basic_timeseries_preprocessing
        self.normalize_floats = self.standardize

        self.use_openfe = self.use_openfe
        self.openfe_kwargs = self.openfe_kwargs
        self.openfe_feature_save_directory = self.openfe_feature_save_directory

        self.specialized_feature_engineering_function = self.specialized_feature_engineering_function
        self.specialized_feature_engineering_function_name = self.get_feature_engineering_function_name()

        self.openfe_name = self.get_openfe_name() if self.use_openfe else ''
        self.name = self.get_name()

    def __call__(self) -> None:
        """Reinitialize class instance when called.

        :return: None
        """
        self.__init__()

    def get_openfe_name(self) -> str:
        """Generates the unique openfe feature name for the current configuration.

        :return: Pipeline name
        """
        openfe_name = sum([str(v) + '_' for k, v in self.openfe_kwargs.items() if k != 'n_jobs'], '')
        openfe_name += f'_{self.use_basic_timeseries_preprocessing}'
        openfe_name += f'_{self.specialized_feature_engineering_function_name}'
        return openfe_name

    def get_name(self) -> str:
        """Concatenates all used preprocessing steps names into a unique pipeline name for each different setup.
        :return: Name for current pipeline setup"""
        setting_names = sum([k + '_' for k, v in self.__dict__.items() if v is True], '')
        return setting_names + self.specialized_feature_engineering_function_name

    def get_feature_engineering_function_name(self) -> str:
        try:
            specialized_feature_engineering_function_name = self.specialized_feature_engineering_function.__name__
        except AttributeError:
            specialized_feature_engineering_function_name = 'None'
        return specialized_feature_engineering_function_name

    def save(self) -> None:
        raise NotImplementedError

    def load(self) -> Any:
        raise NotImplementedError
