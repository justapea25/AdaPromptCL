import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import models
import utils
from config_utils import ConfigHandler

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='AdaPromptCL with YAML configs')
    parser.add_argument('--config', type=str, default='configs/deepfake_basic.yaml',
                       help='Path to YAML config file')
    parser.add_argument('--data_path', type=str, 
                       help='Override data path from config')
    parser.add_argument('--output_dir', type=str,
                       help='Override output directory from config')
    parser.add_argument('--device', type=str,
                       help='Override device from config')
    
    cmd_args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from: {cmd_args.config}")
    config_handler = ConfigHandler(cmd_args.config)
    args = config_handler.args
    
    # Override with command line arguments if provided
    if cmd_args.data_path:
        args.data_path = cmd_args.data_path
    if cmd_args.output_dir:
        args.output_dir = cmd_args.output_dir
    if cmd_args.device:
        args.device = cmd_args.device
    
    # Print configuration
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Experiment: {config_handler.get('experiment_name', 'Unknown')}")
    print(f"Dataset: {args.dataset}")
    print(f"Tasks: {args.deepfake_tasks}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Model: {args.model}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"AdaPromptCL Features:")
    print(f"  - Data Driven Evolve: {args.data_driven_evolve}")
    print(f"  - Uni or Specific: {args.uni_or_specific}")
    print(f"  - Mergable Prompts: {args.mergable_prompts}")
    print("="*60 + "\n")
    
    # Initialize distributed mode
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    
    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # Save config to output directory
        config_handler.save_config(f"{args.output_dir}/config.yaml")
    
    # Build dataloader
    print("Building dataloaders...")
    data_loader, class_mask = build_continual_dataloader(args)
    print(f"Number of tasks: {len(data_loader)}")
    
    for i, task in enumerate(data_loader):
        n_samples = len(task['train'].dataset)
        print(f"Task {i}: {n_samples} training samples")
    
    # Create model
    print(f"Creating model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        args=args
    )
    
    print(f"Total classes: {args.nb_classes}")
    original_model.to(device)
    
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.initializer,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=getattr(args, 'use_g_prompt', False),
        g_prompt_length=getattr(args, 'g_prompt_length', 5),
        g_prompt_layer_idx=getattr(args, 'g_prompt_layer_idx', []),
        use_prefix_tune_for_g_prompt=getattr(args, 'use_prefix_tune_for_g_prompt', True),
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        args=args,
    )
    
    model.to(device)
    
    # DEBUG: Check if binary_classification is properly set
    print(f"Binary classification enabled: {getattr(args, 'binary_classification', 'NOT SET')}")
    
    # DEBUG: Test binary mapping logic
    test_targets = torch.tensor([0, 1, 2, 3])
    test_preds = torch.tensor([0, 1, 2, 3])
    binary_targets = test_targets % 2
    binary_preds = test_preds % 2
    print(f"Test mapping - targets: {test_targets} -> binary: {binary_targets}")
    print(f"Test mapping - preds: {test_preds} -> binary: {binary_preds}")
    print(f"Should all be correct: {(binary_preds == binary_targets).all()}")
    
    # Setup training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Create optimizer
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = create_scheduler(args, optimizer)[0]
    
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    # Train
    train_and_evaluate(model, model_without_ddp, original_model,
                      criterion, data_loader, optimizer, lr_scheduler,
                      device, class_mask, None, args)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    main()