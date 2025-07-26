"""
YAML Configuration Support for AdaPromptCL
"""

import yaml
import argparse
from pathlib import Path

def load_yaml_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def dict_to_namespace(d):
    """Convert dictionary to namespace object recursively"""
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def yaml_to_args(config_path):
    """Convert YAML config to argparse Namespace"""
    config_dict = load_yaml_config(config_path)
    args = dict_to_namespace(config_dict)
    
    # Ensure required attributes exist
    if not hasattr(args, 'distributed'):
        args.distributed = False
    if not hasattr(args, 'device'):
        args.device = 'cuda'
    if not hasattr(args, 'world_size'):
        args.world_size = 1
    if not hasattr(args, 'rank'):
        args.rank = 0
    if not hasattr(args, 'local_rank'):
        args.local_rank = 0
    if not hasattr(args, 'dist_backend'):
        args.dist_backend = 'nccl'
    if not hasattr(args, 'dist_url'):
        args.dist_url = 'env://'
    if not hasattr(args, 'pin_mem'):
        args.pin_mem = True
        
    return args

def validate_deepfake_config(args):
    """Validate deepfake-specific configuration"""
    required_fields = ['task_names', 'data_path']
    for field in required_fields:
        if not hasattr(args, field):
            raise ValueError(f"Missing required field: {field}")
    
    # Check if data path exists
    if not Path(args.data_path).exists():
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
    # Validate task names
    if not args.task_names or len(args.task_names) == 0:
        raise ValueError("task_names cannot be empty")
    
    print(f"✓ Configuration validated successfully")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Tasks: {args.task_names}")
    print(f"  - Number of tasks: {len(args.task_names)}")
    
    return True
