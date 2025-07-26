"""
Main entry point for YAML-based configuration
Supports deepfake datasets while preserving existing functionality
"""

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
from timm.utils.model_ema import ModelEmaV2

# Import both old and new dataset loaders
from datasets import build_continual_dataloader
from continual_datasets.deepfake_datasets import build_deepfake_dataloader, accuracy_binary_deepfake
from yaml_config import yaml_to_args, validate_deepfake_config

from engine import *
import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def flatten_yaml_args(args):
    """Flatten nested YAML arguments to match expected argument structure"""
    flattened = argparse.Namespace()
    
    # Copy all top-level attributes first
    for key, value in vars(args).items():
        if not hasattr(value, '__dict__'):  # Not a nested namespace
            setattr(flattened, key, value)
    
    # Flatten nested namespaces with proper mapping
    if hasattr(args, 'experiment'):
        for key, value in vars(args.experiment).items():
            setattr(flattened, key, value)
    
    if hasattr(args, 'dataset'):
        for key, value in vars(args.dataset).items():
            setattr(flattened, key, value)
    
    if hasattr(args, 'model'):
        for key, value in vars(args.model).items():
            if key == 'name':
                setattr(flattened, 'model', value)
            else:
                setattr(flattened, key, value)
    
    if hasattr(args, 'training'):
        for key, value in vars(args.training).items():
            setattr(flattened, key, value)
    
    if hasattr(args, 'optimizer'):
        for key, value in vars(args.optimizer).items():
            if key == 'type':
                setattr(flattened, 'opt', value)
            elif key == 'eps':
                # Ensure eps is float and map to both names
                eps_val = float(value)
                setattr(flattened, 'eps', eps_val)
                setattr(flattened, 'opt_eps', eps_val)
            elif key == 'betas':
                # Ensure betas is tuple of floats and map to both names
                betas_val = tuple(float(x) for x in value) if isinstance(value, list) else value
                setattr(flattened, 'betas', betas_val)
                setattr(flattened, 'opt_betas', betas_val)
            elif key in ['lr', 'weight_decay', 'momentum', 'clip_grad']:
                # Ensure numeric parameters are floats
                setattr(flattened, key, float(value))
            else:
                setattr(flattened, key, value)
    
    if hasattr(args, 'scheduler'):
        for key, value in vars(args.scheduler).items():
            if key == 'type':
                setattr(flattened, 'sched', value)
            elif key in ['lr', 'min_lr', 'warmup_lr', 'decay_rate']:
                # Ensure learning rate parameters are floats
                setattr(flattened, key, float(value))
            else:
                setattr(flattened, key, value)
    
    if hasattr(args, 'continual_learning'):
        for key, value in vars(args.continual_learning).items():
            setattr(flattened, key, value)
    
    if hasattr(args, 'prompt'):
        for key, value in vars(args.prompt).items():
            # Map YAML names to expected argument names
            if key == 'size':
                setattr(flattened, 'size', value)  # pool_size
            elif key == 'length':
                setattr(flattened, 'length', value)  # prompt_length
            else:
                setattr(flattened, key, value)
    
    if hasattr(args, 'model_advanced'):
        for key, value in vars(args.model_advanced).items():
            setattr(flattened, key, value)
    
    if hasattr(args, 'augmentation'):
        for key, value in vars(args.augmentation).items():
            setattr(flattened, key, value)
    
    if hasattr(args, 'system'):
        for key, value in vars(args.system).items():
            setattr(flattened, key, value)
    
    # Ensure critical attributes are set
    if not hasattr(flattened, 'model'):
        raise ValueError("Model name not specified in configuration")
    
    return flattened

def set_comprehensive_default_args(args):
    """Set comprehensive default values for ALL arguments expected by AdaPromptCL"""
    
    # Comprehensive defaults based on cifar100_dualprompt.py and engine.py analysis
    comprehensive_defaults = {
        # Core system settings
        'device': 'cuda',
        'distributed': False,
        'world_size': 1,
        'rank': 0,
        'local_rank': 0,
        'dist_backend': 'nccl',
        'dist_url': 'env://',
        'gpu': 0,
        'print_freq': 10,
        
        # Training basics
        'epochs': 10,
        'batch_size': 32,
        'num_workers': 4,
        'pin_mem': True,
        
        # Model basics
        'model': 'vit_base_patch16_224',
        'input_size': 224,
        'pretrained': True,
        'drop': 0.0,
        'drop_path': 0.0,
        
        # Optimizer settings
        'opt': 'adam',
        'lr': 0.03,
        'weight_decay': 0.0,
        'eps': 1e-8,
        'opt_eps': 1e-8,
        'betas': (0.9, 0.999),
        'opt_betas': (0.9, 0.999),
        'momentum': 0.9,
        'clip_grad': 1.0,
        'reinit_optimizer': True,
        
        # Scheduler settings
        'sched': 'constant',
        'warmup_lr': 1e-6,
        'min_lr': 1e-5,
        'decay_epochs': 30,
        'warmup_epochs': 5,
        'cooldown_epochs': 10,
        'patience_epochs': 10,
        'decay_rate': 0.1,
        'scale_lr': True,
        'lr_noise': None,
        'lr_noise_pct': 0.67,
        'lr_noise_std': 1.0,
        
        # Dataset and CL settings
        'dataset': 'unknown',
        'data_path': '',
        'num_tasks': 0,
        'nb_classes': 0,
        'task_inc': False,
        'train_mask': False,
        
        # CLIP settings
        'clip_emb': False,
        'clip_text_head': False,
        'text_img_prob': 0.5,
        'freeze_clipproj': False,
        'freeze_clipproj_t': 1,
        
        # Task and specialization
        'task_agnostic_head': False,
        'sep_specialization': False,
        'eval_known_prompt': False,
        'sep_criterion': 'random',
        'wo_sep_mask': False,
        'task_lvl_sep_mask': False,
        'wo_sep_mask_e': -1,
        'w_sep_mask_e': -1,
        'wo_sep_mask_t': -1,
        'w_sep_mask_t': -1,
        
        # Core prompt settings
        'prompt_pool': True,
        'size': 10,
        'length': 8,
        'top_k': 1,
        'batchwise_prompt': True,
        'prompt_key': True,
        'prompt_key_init': 'uniform',
        'initializer': 'uniform',
        'embedding_key': 'cls',
        'predefined_key': '',
        'use_prompt_mask': True,
        'mask_first_epoch': False,
        'pull_constraint': True,
        'pull_constraint_coeff': 1.0,
        'same_key_value': False,
        'shared_prompt_pool': False,
        'shared_prompt_key': False,
        'shared_mean': False,
        'prompt_same_init': True,
        'prompt_momentum': 0.01,
        
        # G-prompt settings
        'use_g_prompt': False,
        'g_prompt_length': 1,
        'g_prompt_layer_idx': [0],
        'use_prefix_tune_for_g_prompt': False,
        
        # E-prompt settings
        'use_e_prompt': False,
        'e_prompt_layer_idx': [0],
        'use_prefix_tune_for_e_prompt': False,
        
        # Advanced prompt features
        'coda_prompt': False,
        'num_coda_prompt': 10,
        'feat_prompt': False,
        'num_feat_prompt': 100,
        'orthogonal_coeff': 1.0,
        'softmax_prompt': False,
        'normalize_prompt': False,
        
        # Specific prompts
        'specific_prompts': False,
        's_prompts_layer_idx': [],
        's_prompts_num_prompts': 5,
        's_prompts_length': 1,
        's_prompts_add': 'cat',
        
        # Advanced features
        'keep_prompt_freq': False,
        'feat_attn_mask': False,
        'copy_top2bottom': False,
        'task_free': False,
        'copy_thrh': 0.05,
        'pct_copy': 0.1,
        'compress_ratio': 1,
        'top_ratio': 1.0,
        
        # Waiting room
        'wr_prompt': False,
        'num_wr_prompt': 0,
        
        # Supervised contrastive learning
        'supcon_prompts': False,
        'supcon_attns': False,
        'loss_worldsz': False,
        'prev_cls_prompts': False,
        'prev_cls_attns': False,
        'not_use_qnet': False,
        'not_use_qnet_attns': False,
        'q_units': 500,
        'q_units_attns': 500,
        'supcon_temp': 0.1,
        'supcon_temp_attns': 0.1,
        'coef_supcon': 0.1,
        'coef_supcon_attns': 0.1,
        'prompt_ema_ratio': 0.9,
        'attns_ema_ratio': 0.9,
        'partial_layers': [],
        'agg_dims': [],
        'anneal_supcon': False,
        'func_anneal': 'linear',
        'dest_coef_supcon': 0.1,
        
        # Contrastive learning
        'cl_prompts': False,
        'cl_randaug': False,
        
        # Superclass prompts
        'supcls_prompts': False,
        'supcls_eval_known': False,
        'shared_prompt': False,
        'sep_shared_prompt': False,
        'only_shared_prompt': False,
        'one_shared_prompt': False,
        'shared_prompt_ema_ratio': 0.9,
        
        # Training settings
        'num_updates': 1,
        'grad_accum': False,
        'accum_step': 1,
        
        # Model head settings
        'head_type': 'token',
        'eval_prototype_clf': False,
        'prototype_ema_ratio': 0.9,
        
        # Model variations
        'pt_augmented_ptm': False,
        'ptm_load_msd': False,
        'load_dir': 'sth',
        'ptm_num_feat_prompt': 10,
        
        # Data-driven evolution
        'data_driven_evolve': False,
        'uni_or_specific': False,
        'wo_normalizing_prompts': False,
        'compare_after_warmup': False,
        'evolve_epoch': 2,
        'left_1Tepochs': 0,
        
        # FiLM settings
        'film_train': False,
        'nofilm': False,
        'film_combine': 'mean',
        'film_train_epoch_add': False,
        'film_train_epoch': 10,
        'use_film': False,
        
        # Merging settings
        'avg_merge': False,
        'ema_merge': False,
        'postmerge_thrh': 0.5,
        'mergable_prompts': False,
        'mergable_cands': None,
        'spcf_epochs': 100,
        'uni_epochs': 5,
        'mergable_epochs': 5,
        'max_mergable_epochs': 50,
        'reinit_clf': True,
        'sampling_size_for_remerge': 1000,
        'save_prepgroup_info': False,
        'kmeans_run': 100,
        'kmeans_top_n': 2,
        'converge_thrh': 0.3,
        'bidirectional': False,
        'normalize_by_sim': False,
        'mix_with_sim': False,
        'bwd_ratio': 0.3,
        'fwd_ratio': 0.3,
        
        # Saving settings
        'save_sdict': False,
        'save_sdict_by_epochs': False,
        'save_sdict_epochs': [],
        'save_warmup_prompts': False,
        
        # VTAB settings
        'milder_vtab': False,
        'no_mild_tasks': False,
        'num_no_mild_tasks': 8,
        'clustering_based_vtabgroups': False,
        'vtab_group_order': False,
        'milder_by_alldts': False,
        'overlap_similarity': 50,
        'vtab_datasets': [],
        
        # Overlapping tasks
        'num_overlaps': 1,
        'num_overlapping_tasks': 1,
        'shuffle_overlaps': False,
        'shuffle_overlaps_inds': [],
        'shuffle_prompt_inds': [],
        'overlap_dataset_scale': 0.5,
        'overlap_datasets': [],
        
        # Freezing settings
        'freeze': [],
        'freezeE': False,
        'freezeE_zerograd': False,
        'freezeE_t': -2,
        'freezeG': False,
        'freezeG_t': 1,
        'freezeG_e': -1,
        'freezeH': False,
        'freezeH_t': -2,
        'freezeQ': False,
        'freezeV': False,
        'freezeK': False,
        
        # Model merging
        'merge_pt': False,
        
        # Noisy settings
        'noisy_task_boundary': False,
        'noisy_pct': 0.1,
        'noisy_type': 'start',
        'noisy_policy': '1-prev',
        'noisy_labels': False,
        'in_task_noise': False,
        'noisy_labels_type': 'symmetry',
        'noisy_labels_rate': 0.1,
        
        # Small dataset
        'small_dataset': False,
        'small_dataset_scale': 1.0,
        
        # Blurry CL
        'blurryCL': False,
        'blurryM': 0.9,
        'balanced_softmax': False,
        'online_cls_mask': False,
        
        # Evaluation settings
        'eval_only_acc': False,
        'anytime_inference': False,
        'anytime_inference_period': 100,
        
        # Augmentation settings
        'color_jitter': None,
        'aa': None,
        'smoothing': 0.1,
        'train_interpolation': 'bicubic',
        'reprob': 0.0,
        'remode': 'pixel',
        'recount': 1,
        
        # Model EMA
        'model_ema': False,
        'model_ema_decay': 0.9999,
        'model_ema_force_cpu': False,
        
        # Additional attributes that might be referenced
        'exposed_classes': set(),
        'uni_or_spcf_gt': [],
        'task_id_list': [],
        'only_G': False,
        'output_dir': './output',
        'seed': 42,
    }
    
    # Set defaults for any missing attributes
    for key, default_value in comprehensive_defaults.items():
        if not hasattr(args, key):
            setattr(args, key, default_value)
    
    # Type conversion for critical parameters
    numeric_params = ['eps', 'opt_eps', 'lr', 'warmup_lr', 'min_lr', 'weight_decay', 
                     'momentum', 'decay_rate', 'clip_grad']
    for param in numeric_params:
        if hasattr(args, param) and args.__dict__[param] is not None:
            setattr(args, param, float(getattr(args, param)))
    
    # Ensure betas is a tuple of floats
    for param in ['betas', 'opt_betas']:
        if hasattr(args, param) and getattr(args, param) is not None:
            val = getattr(args, param)
            if isinstance(val, list):
                setattr(args, param, tuple(float(x) for x in val))
            elif not isinstance(val, tuple):
                setattr(args, param, (0.9, 0.999))
    
    return args

def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # Choose dataloader based on dataset type
    if hasattr(args, 'type') and args.type == 'deepfake':
        print("Using deepfake dataloader...")
        from continual_datasets.deepfake_datasets import build_deepfake_dataloader
        data_loader, class_mask = build_deepfake_dataloader(args)
    else:
        print("Using standard dataloader...")
        data_loader, class_mask = build_continual_dataloader(args)
    
    print(f"Number of tasks: {len(data_loader)}")
    print(f"Number of classes: {args.nb_classes}")

    if args.nb_classes == 0:
        raise ValueError("Number of classes is 0. Check your dataset configuration.")

    # CRITICAL: Set all comprehensive default arguments before anything else
    args = set_comprehensive_default_args(args)
    
    print("✅ All default arguments set successfully!")

    # Create original model (required by AdaPromptCL)
    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes if not args.clip_text_head else 512,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        args=args
    )

    # Create main model with all prompt parameters
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes if not args.clip_text_head else 512,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        # Prompt-related parameters
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        # G-prompt parameters
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        # E-prompt parameters  
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        # Other parameters
        same_key_value=args.same_key_value,
        eval_prototype_clf=args.eval_prototype_clf,
        args=args
    )

    # Move to device
    original_model.to(device)
    model.to(device)

    # Handle freezing if specified
    if args.freeze:
        # Freeze original model parameters
        for p in original_model.parameters():
            p.requires_grad = False
        
        # Freeze specified parts of main model
        if args.clip_text_head:
            args.freeze += ['head']
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    # Create optimizer  
    optimizer = create_optimizer(args, model)

    # Create scheduler
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Model EMA
    ema_model = None
    if args.model_ema:
        ema_model = ModelEmaV2(model, decay=args.model_ema_decay)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Create criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # Train and evaluate
    train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, ema_model, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('YAML-based AdaPromptCL training and evaluation')
    parser.add_argument('--config', type=str, default='configs/deepfake_cddb_dualprompt.yaml', 
                       help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Load YAML configuration
    print(f"Loading configuration from: {args.config}")
    yaml_args = yaml_to_args(args.config)
    
    # Flatten nested arguments
    flattened_args = flatten_yaml_args(yaml_args)
    
    # Validate configuration if it's a deepfake experiment
    if hasattr(flattened_args, 'type') and flattened_args.type == 'deepfake':
        validate_deepfake_config(flattened_args)
    
    # Create output directory
    if flattened_args.output_dir:
        Path(flattened_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Configuration loaded successfully!")
    print(f"Experiment: {getattr(flattened_args, 'name', 'unnamed')}")
    print(f"Output directory: {flattened_args.output_dir}")
    
    main(flattened_args)
    
    sys.exit(0)
