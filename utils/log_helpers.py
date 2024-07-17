import logging
def get_logger(name: str, base_filepath: str = 'logs/model_experiments') -> logging.Logger:
    """Creates a logging.Logger object that writes a logfile named name.log into the folder at base_filepath
    (throws an error if the folder does not exist)

    :param name: Name of the logger and the logging file
    :param base_filepath: Path to the folder in which the logging file will get saved
    :return: Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'{base_filepath}/{name}.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger