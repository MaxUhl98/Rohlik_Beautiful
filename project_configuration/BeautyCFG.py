class BeautyCFG:
    """Configuration class to control beauty checkup for Rohlik_Beautiful project"""
    project_path: str = r'D:\PycharmProjects\Rohlik_Beautiful'

    # Analysis configuration
    beauty_checkup_logging_directory: str = project_path + '/logs/maintainability'
    coverage_log_file: str = project_path + '/logs/coverage/coverage.log'
    analysis_excluded_dirs: list[str] = ['.venv', '__pycache__', '.git', '.idea']
    analysis_excluded_files: list[str] = ['playground.py', '__init__.py']
