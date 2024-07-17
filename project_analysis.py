from radon.complexity import ComplexityVisitor, cc_rank
from radon.metrics import mi_visit, h_visit, halstead_visitor_report, mi_rank
from typing import *
import os
import logging
from utils.log_helpers import get_logger
from pprint import pformat


def analyze_script(script_path: Union[str, os.PathLike], logger: logging.Logger) -> None:
    with open(script_path, 'r') as f:
        code = f.read()
    visitor = ComplexityVisitor.from_code(code)
    scores = [(cc_rank(func.complexity), func.name) for func in visitor.functions]
    scores = list(filter(lambda s: s[0] in ['C', 'D', 'E', 'F'], scores))
    logger.info(f'Problematic Cyclomatic Complexities:\n{pformat(scores)}')
    v = h_visit(code)
    halsted_scores = [(func.difficulty, func.bugs, name) for name, func in v.functions]
    halsted_scores = list(filter(lambda x: x[0] >= 10 or x[1] >= 1, halsted_scores))
    logger.info(f'Problematic Halstead Scores:\n{pformat(halsted_scores)}')
    if mi_visit(code, False) < 20:
        logger.info(f'Code at {script_path} is not easily maintainable')


if __name__ == '__main__':
    script_path = 'playground.py'
    analyze_script(script_path, get_logger(script_path.replace('.py', ''), 'logs/maintainability'))
