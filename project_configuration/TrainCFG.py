from ast import literal_eval
from typing_extensions import Self
from project_configuration.ConfigurationFunctionalities.BasicFunctionalities import BasicFunctionalities

class TrainCFG(BasicFunctionalities):
    """Class to configure the training process"""
    run_name: str = 'test'
    validation_time_steps: int = 60
    num_folds: int = 10
    run_save_directory: str = 'runs/'
    log_dir: str = 'logs/training'
    aggregrate_runs_path: str = 'runs/aggregate_runs.csv'

    def __init__(self):
        """This is necessary for the saving of variables
        (__dict__ values do not get set for class attributes)"""
        super().__init__()
        self.run_name = self.run_name
        self.validation_time_steps = self.validation_time_steps
        self.num_folds = self.num_folds
        self.run_save_directory = self.run_save_directory
        self.log_dir = self.log_dir


