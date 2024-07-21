from ast import literal_eval
from typing_extensions import Self


class TrainCFG:
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
        self.run_name = self.run_name
        self.validation_time_steps = self.validation_time_steps
        self.num_folds = self.num_folds
        self.run_save_directory = self.run_save_directory
        self.log_dir = self.log_dir

    def save(self, save_path: str) -> None:
        """Saves the current configuration setup to a txt file at a given location

        :param save_path: Destination path to save the configuration setup to
        :return: None
        """
        with open(save_path, 'w') as f:
            f.write(str({k: v for k, v in vars(self).items() if not callable(v) and not k.startswith("__")}))

    def load(self, load_path: str) -> Self:
        """Loads the configuration setup from a txt file at a given location

        :param load_path: Path to the configuration setup txt file
        :return: TrainCFG object with loaded configuration
        """
        with open(load_path, 'r') as f:
            self.__dict__.update(literal_eval(f.read()))
        return self

    def __eq__(self, other) -> bool:
        """Compares if two TrainCFG objects are equal

        :param other: Other TrainCFG object to compare current object to
        :return: True if both have equal attributes, False otherwise
        """
        return self.__dict__ == other.__dict__
