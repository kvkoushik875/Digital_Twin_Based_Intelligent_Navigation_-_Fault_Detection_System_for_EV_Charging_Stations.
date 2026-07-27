import logging
import logging.config
import yaml
import os

def load_logging():
    def load_logging():
        """
        Loads logging configuration from config/logging.conf
        and ensures logs directory exists at project root.
        """

        # FIX: go up 3 levels to reach project root
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )

        # Ensure logs directory exists
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Build correct path to logging.conf
        config_path = os.path.join(project_root, "config", "logging.conf")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Logging config not found at: {config_path}")

        logging.config.fileConfig(config_path)


def get_logger(name: str):
    """
    Returns a logger instance for the given module name.
    """
    return logging.getLogger(name)

# Initialize logging on import
load_logging()

# Common loggers used across the project
system_logger = get_logger("system")
mqtt_logger = get_logger("mqtt")
model_logger = get_logger("model")
digital_twin_logger = get_logger("digital_twin")
navigation_logger = get_logger("navigation")
