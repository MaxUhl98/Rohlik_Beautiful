class TrainCFG:
    """Class to configure the training process"""
    run_name: str = 'test'

    validation_time_steps: int = 60
    num_folds: int = 10

    run_save_directory: str = 'runs/'

    def save(self, save_path: str) -> None:
        with open(save_path, 'w') as f:
            f.write(str({k: v for k, v in vars(self).items() if not callable(v) and not k.startswith("__")}))
