class CFG:
    """
    Configuration class for Rohlik_Beautiful project
    """
    project_path = r'D:\PycharmProjects\Rohlik_Beautiful'

    # Analysis configuration
    beauty_checkup_logging_directory = project_path + '/logs/maintainability'
    analysis_excluded_dirs = ['.venv', '__pycache__', '.git', '.idea']
    analysis_excluded_files = ['playground.py', '__init__.py']

