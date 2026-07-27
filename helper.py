import yaml
import os

def load_settings():
    """
    Loads YAML configuration settings from config/settings.yaml
    regardless of where the script is executed.
    """

    # Project root is 3 levels above this file
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )

    settings_path = os.path.join(project_root, "config", "settings.yaml")

    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"Settings file not found at: {settings_path}")

    with open(settings_path, "r") as f:
        return yaml.safe_load(f)
