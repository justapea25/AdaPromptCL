import yaml
import argparse
from typing import Dict, Any
from pathlib import Path

class ConfigHandler:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.args = self.config_to_args()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def config_to_args(self):
        """Convert config dict to argparse Namespace for compatibility"""
        args = argparse.Namespace()
        
        # Set all config values as attributes
        for key, value in self.config.items():
            setattr(args, key.replace('-', '_'), value)
        
        # Ensure required attributes exist with defaults
        self._set_all_defaults(args)
        
        return args
    
    def _set_all_defaults(self, args):
        """Set ALL default values from original configs to avoid missing attributes"""
        
        # Complete list of ALL parameters from original configs
        all_defaults = {
            # Basic training parameters
            'batch_size': 32,
            'epochs': 5,
            'num_workers': 4,
            'pin_mem': True,
            'eval': False,
            'seed': 42,
            'print_freq': 10,
            
            # Model parameters
            'model': 'vit_base_patch16_224',
            'input_size': 224,
            'pretrained': True,
            'drop': 0.0,
            'drop_path': 0.0,
            
            # Optimizer parameters
            'opt': 'adam',
            'opt_eps': 1e-8,
            'opt_betas': [0.9, 0.999],
            'clip_grad': 1.0,
            'momentum': 0.9,
            'weight_decay': 0.0,
            'reinit_optimizer': True,
            
            # Learning rate parameters
            'sched': 'constant',
            'lr': 0.01,
            'lr_noise': None,
            'lr_noise_pct': 0.67,
            'lr_noise_std': 1.0,
            'warmup_lr': 1e-6,
            'min_lr': 1e-5,
            'decay_epochs': 30,
            'warmup_epochs': 5,
            'cooldown_epochs': 10,
            'patience_epochs': 10,
            'decay_rate': 0.1,
            'scale_lr': True,
            
            # Augmentation parameters
            'color_jitter': None,
            'aa': None,
            'smoothing': 0.1,
            'train_interpolation': 'bicubic',
            'reprob': 0.0,
            'remode': 'pixel',
            'recount': 1,
            
            # Data parameters
            'shuffle': False,
            'small_dataset': False,
            'small_dataset_scale': 1.0,
            
            # System parameters
            'device': 'cuda',
            'distributed': False,
            'world_size': 1,
            'dist_url': 'env://',
            'dist_backend': 'nccl',
            
            # Continual learning parameters
            'task_inc': False,
            'train_mask': True,
            
            # G-Prompt parameters
            'use_g_prompt': False,
            'g_prompt_length': 5,
            'g_prompt_layer_idx': [],
            'use_prefix_tune_for_g_prompt': True,
            
            # E-Prompt parameters
            'use_e_prompt': False,
            'e_prompt_layer_idx': [],
            'use_prefix_tune_for_e_prompt': True,
            
            # Prompt pool parameters
            'prompt_pool': True,
            'size': 100,
            'length': 5,
            'top_k': 5,
            'initializer': 'uniform',
            'prompt_key': True,
            'prompt_key_init': 'uniform',
            'use_prompt_mask': True,
            'mask_first_epoch': False,
            'shared_prompt_pool': False,
            'shared_prompt_key': False,
            'batchwise_prompt': False,
            'embedding_key': 'cls',
            'predefined_key': '',
            'pull_constraint': True,
            'pull_constraint_coeff': 1.0,
            'same_key_value': False,
            
            # CODA-prompt
            'coda_prompt': False,
            'num_coda_prompt': 10,
            
            # Feature prompt
            'feat_prompt': False,
            'num_feat_prompt': 100,
            'orthogonal_coeff': 1.0,
            'softmax_prompt': False,
            'save_attn_scores': False,
            'save_attn_period': 2,
            'keep_prompt_freq': False,
            'feat_attn_mask': False,
            
            # Copy top2bottom
            'copy_top2bottom': False,
            'task_free': False,
            'copy_thrh': 0.05,
            'pct_copy': 0.1,
            'compress_ratio': 1,
            'top_ratio': 1.0,
            'freezeQ': False,
            'freezeV': False,
            'freezeK': False,
            
            # Waiting room
            'wr_prompt': False,
            'num_wr_prompt': 0,
            
            # Supervised contrastive
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
            
            # Contrastive loss
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
            'num_updates': 1,
            
            # Original model
            'pt_augmented_ptm': False,
            'ptm_load_msd': False,
            'load_dir': 'sth',
            'ptm_num_feat_prompt': 10,
            'normalize_prompt': False,
            
            # Specific prompts
            'specific_prompts': False,
            's_prompts_layer_idx': [],
            's_prompts_num_prompts': 5,
            's_prompts_length': 1,
            's_prompts_add': 'cat',
            
            # Evolve prompts / AdaPromptCL
            'data_driven_evolve': False,
            'uni_or_specific': False,
            'wo_normalizing_prompts': False,
            'compare_after_warmup': False,
            'evolve_epoch': 2,
            'left_1Tepochs': 0,
            'converge_thrh': 0.4,
            'bidirectional': False,
            'normalize_by_sim': False,
            'mix_with_sim': False,
            'bwd_ratio': 0.3,
            'fwd_ratio': 0.3,
            
            # Film
            'film_train': False,
            'nofilm': False,
            'film_combine': 'mean',
            'film_train_epoch_add': False,
            'film_train_epoch': 10,
            
            # Merge
            'avg_merge': False,
            'ema_merge': False,
            'postmerge_thrh': 0.6,
            'mergable_prompts': False,
            'spcf_epochs': 100,
            'uni_epochs': 5,
            'mergable_epochs': 5,
            'max_mergable_epochs': 50,
            'reinit_clf': True,
            'sampling_size_for_remerge': 1000,
            'save_prepgroup_info': False,
            'kmeans_run': 100,
            'kmeans_top_n': 2,
            
            # Save state
            'save_sdict': False,
            'save_sdict_by_epochs': False,
            'save_sdict_epochs': [],
            'save_warmup_prompts': False,
            
            # VTAB similar tasks
            'milder_vtab': False,
            'no_mild_tasks': False,
            'num_no_mild_tasks': 8,
            'clustering_based_vtabgroups': False,
            'vtab_group_order': False,
            'milder_by_alldts': False,
            'overlap_similarity': 50,
            'num_overlaps': 1,
            'num_overlapping_tasks': 1,
            'shuffle_overlaps': False,
            'shuffle_overlaps_inds': [],
            'shuffle_prompt_inds': [],
            'overlap_dataset_scale': 0.5,
            'overlap_datasets': [],
            'prompt_same_init': True,
            
            # Prototype classifier
            'eval_prototype_clf': False,
            'prototype_ema_ratio': 0.9,
            
            # Freeze
            'freezeE': False,
            'freezeE_zerograd': False,
            'freezeE_t': -2,
            'freezeE_e': -2,
            'unfreezeE': False,
            'unfreezeE_e': -2,
            'merge_pt': False,
            'freezeH': False,
            'freezeH_t': -2,
            
            # CLIP embedding
            'clip_emb': False,  # THIS WAS MISSING!
            'text_img_prob': 0.5,
            'grad_accum': False,
            'accum_step': 1,
            'freezeG': False,
            'freezeG_t': 1,
            'freezeG_e': -1,
            'freeze_clipproj': False,
            'freeze_clipproj_t': 1,
            
            # Task agnostic head
            'task_agnostic_head': False,
            'clip_text_head': False,
            
            # Noisy task boundary
            'noisy_task_boundary': False,
            'noisy_pct': 0.1,
            'noisy_type': 'start',
            'noisy_policy': '1-prev',
            
            # Noisy labels
            'noisy_labels': False,
            'in_task_noise': False,
            'noisy_labels_type': 'symmetry',
            'noisy_labels_rate': 0.1,
            
            # Blurry CL
            'blurryCL': False,
            'blurryM': 0.9,
            'balanced_softmax': False,
            'online_cls_mask': False,
            
            # Separation
            'sep_specialization': False,
            'sep_criterion': 'random',
            'wo_sep_mask': False,
            'task_lvl_sep_mask': False,
            'wo_sep_mask_e': -1,
            'w_sep_mask_e': -1,
            'wo_sep_mask_t': -1,
            'w_sep_mask_t': -1,
            
            # Evaluation
            'eval_known_prompt': False,
            'eval_only_acc': False,
            'anytime_inference': False,
            'anytime_inference_period': 4,
            'loss_noise_pair': False,
            'only_G': False,
            
            # ViT parameters
            'global_pool': 'token',
            'head_type': 'token',
            'freeze': ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'],
            
            # VTAB datasets
            'vtab_datasets': [],
            'binary_classification': False,
        }
        
        # Set all defaults
        for key, default_value in all_defaults.items():
            if not hasattr(args, key):
                setattr(args, key, default_value)
    
    def __getitem__(self, key: str):
        return self.config[key]
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def save_config(self, save_path: str):
        """Save current config to file"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
