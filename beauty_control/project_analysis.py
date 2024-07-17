from radon.complexity import ComplexityVisitor, cc_rank
from radon.metrics import mi_visit, h_visit
from typing import *
import os
import logging
from utils.log_helpers import get_logger
from pprint import pformat
from CFG import CFG
from pathlib import Path


def check_cyclonic_complexity(code: str, script_name: str, logger: logging.Logger) -> None:
    visitor = ComplexityVisitor.from_code(code)
    scores = [(cc_rank(func.complexity), func.name) for func in visitor.functions]
    problematic_cyclonic_complexity_scores = list(filter(lambda s: s[0] in ['C', 'D', 'E', 'F'], scores))

    if problematic_cyclonic_complexity_scores:
        logger.info(f'Problematic Cyclomatic Complexities in file {script_name}:')
        for problem in problematic_cyclonic_complexity_scores:
            logger.info(f'Function {problem[1]} has complexity rank {problem[0]}')


def check_halstead_complexity(code: str, script_name: str, logger: logging.Logger) -> None:
    v = h_visit(code)
    halsted_scores = [(func.difficulty, func.bugs, name) for name, func in v.functions]
    problematic_halsted_scores = list(filter(lambda x: x[0] >= 10 or x[1] >= 1, halsted_scores))
    if problematic_halsted_scores:
        logger.info(f'Problematic Halstead Scores in file {script_name}:')
        for problem in problematic_halsted_scores:
            logger.info(f'Function {problem[-1]}, Difficulty: {problem[0]}, Bugs: {problem[1]}')


def analyze_script(script_path: Union[str, os.PathLike], logger: logging.Logger) -> None:
    script_name = script_path.rsplit("\\", 1)[1]
    with open(script_path, 'r') as f:
        code = f.read()

    check_cyclonic_complexity(code, script_name, logger)
    check_halstead_complexity(code, script_name, logger)
    if mi_visit(code, False) < 20:
        logger.info(f'Code at {script_path} is not easily maintainable')


def get_whole_project_analysis() -> None:
    all_files = Path(CFG.project_path).glob('**/*.py')
    all_files = list(map(lambda x: str(x.absolute()), all_files))
    all_files = list(filter(
        lambda f: not any([excluded in f for excluded in CFG.analysis_excluded_dirs + CFG.analysis_excluded_files]),
        all_files))
    for file in all_files:
        file_beauty_logger = get_logger(file.rsplit('\\', maxsplit=1)[1].replace('.py', ''),
                                        base_filepath=CFG.beauty_checkup_logging_directory)
        analyze_script(file, file_beauty_logger)


if __name__ == '__main__':
    get_whole_project_analysis()
