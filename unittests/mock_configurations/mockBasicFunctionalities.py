from ast import literal_eval
from typing import *


class MockBasicFunctionalities:

    def save(self, save_path: str) -> None:
        """Saves the current configuration setup to a txt file at a given location

        :param save_path: Destination path to save the configuration setup to
        :return: None
        """
        with open(save_path, 'w') as f:
            f.write(str({k: v for k, v in vars(self).items() if not callable(v) and not k.startswith("__")}))

    def load(self, load_path: str) -> Any:
        """Loads the configuration setup from a txt file at a given location

        :param load_path: Path to the configuration setup txt file
        :return: TrainCFG object with loaded configuration
        """
        with open(load_path, 'r') as f:
            self.__dict__.update(literal_eval(f.read()))
        return self

    def __eq__(self, other) -> bool:
        """Compares if two objects are equal

        :param other: Other object to compare current object to
        :return: True if both have equal attributes, False otherwise
        """
        return self.__dict__ == other.__dict__

    def __call__(self) -> None:
        """Reinitialize class instance when called.

        :return: None
        """
        self.__init__()
