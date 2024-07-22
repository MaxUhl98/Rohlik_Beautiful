from project_configuration import PreprocessingCFG, DataCFG


class BasePipeline:
    """Base class for preprocessing pipelines"""

    def __init__(self, preprocess_cfg: PreprocessingCFG, data_cfg: DataCFG):
        self.preprocess_cfg = preprocess_cfg
        self.data_cfg = data_cfg
