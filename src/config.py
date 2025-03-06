from yacs.config import CfgNode as CN

def load_config(config_file):
    """
    Load a configuration file into a CfgNode.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        CN: The loaded configuration as a CfgNode object.
    """
    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    return cfg

# Load individual configurations from YAML files
cfg_logger = load_config("settings/logger.yml")

# Initialize the main configuration and allow new keys
cfg = CN()
cfg.set_new_allowed(True)

# Merge individual configurations into their respective sections in the main config
cfg.logger = CN()
cfg.logger.set_new_allowed(True)
cfg.logger.merge_from_other_cfg(cfg_logger)