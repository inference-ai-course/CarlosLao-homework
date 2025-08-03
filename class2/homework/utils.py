import os


def get_path(output_folder):
    """
    Returns an absolute path to the output folder, relative to the script location.
    Creates the folder if it doesn't exist.
    """
    try:
        # Works when running as a script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for interactive environments (e.g., Jupyter, REPL)
        script_dir = os.getcwd()

    output_path = os.path.join(script_dir, output_folder)
    return output_path
