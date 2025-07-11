import yaml
import os
from string import Template

def load_config(path, default_path=None, loaded_files=None):
    """
    Loads a config file with optional inheritance.

    Args:
        path (str): Path to the main config file.
        default_path (str, optional): Path to the default config file if inheritance is not specified.
        loaded_files (set, optional): Tracks loaded files to prevent circular inheritance.

    Returns:
        dict: Combined configuration dictionary.
    """
    if loaded_files is None:
        loaded_files = set()

    if path in loaded_files:
        raise ValueError(f"Circular inheritance detected for {path}")

    # Track the loaded file to detect any circular inheritance
    loaded_files.add(path)

    # Load the main config file
    with open(path, "r") as f:
        cfg_special = yaml.full_load(f)

    # Handle inheritance
    inherit_from = cfg_special.get("inherit_from")
    if inherit_from:
        if isinstance(inherit_from, list):
            # If inherit_from is a list, load each file and merge them
            cfg = {}
            for inherit_path in inherit_from:
                inherited_cfg = load_config(inherit_path, default_path, loaded_files)
                update_recursive(cfg, inherited_cfg)
        else:
            # If inherit_from is a string, load the single inherited config
            cfg = load_config(inherit_from, default_path, loaded_files)
    elif default_path:
        # Load the default config if specified
        with open(default_path, "r") as f:
            cfg = yaml.full_load(f)
    else:
        cfg = {}

    # Merge main configuration with inherited configuration
    update_recursive(cfg, cfg_special)
    return cfg

def update_recursive(dict1, dict2):
    """
    Recursively updates dict1 with dict2, where dict2 values take precedence.

    Args:
        dict1 (dict): The dictionary to be updated.
        dict2 (dict): The dictionary with updates.
    """
    for k, v in dict2.items():
        if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def save_config(config, output_path):
    """
    Saves the configuration dictionary to a YAML file.

    Args:
        config (dict): Configuration dictionary to save.
        output_path (str): Path to save the YAML file.
    """
    output_path = output_path+'/.configs'
    os.makedirs(output_path, exist_ok=True)
    with open(output_path+'/config.yaml', 'w') as f:
        yaml.dump(config, f)


