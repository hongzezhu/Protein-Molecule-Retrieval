import sys
import os
import copy

sys.path.append('.')

import yaml
import argparse

from easydict import EasyDict
from utils.others import setup_seed
from utils.module_loader import *


def run(config):
    # Initialize a model
    model = load_model(config.model)

    # Initialize a dataset
    data_module = load_dataset(config.dataset)

    # Check if phased training is enabled
    phased_training = config.get("phased_training", {})
    enable_phased = phased_training.get("enable", False)
    
    if enable_phased:
        # Phased training: LiT (Locked-image Tuning) strategy
        print("\n" + "="*80)
        print("PHASED TRAINING ENABLED - LiT Strategy")
        print("="*80)
        
        # Phase 1: Projection Warmup
        phase1_epochs = phased_training.get("phase1_epochs", 2)
        # Create a copy of config for phase 1
        import copy
        phase1_config = EasyDict(copy.deepcopy(dict(config)))
        phase1_config.Trainer.max_epochs = phase1_epochs
        
        print(f"\n[Phase 1] Projection Warmup: {phase1_epochs} epochs")
        print("  - Freezing: Protein Tower, Molecular Tower")
        print("  - Training: Projection Layers only")
        print("-"*80)
        
        model.set_training_phase("projection_warmup")
        trainer1 = load_trainer(phase1_config)
        trainer1.fit(model=model, datamodule=data_module)
        
        # Phase 2: Molecule Training
        total_epochs = config.Trainer.max_epochs
        phase2_epochs = phased_training.get("phase2_epochs", total_epochs - phase1_epochs)
        # Create a copy of config for phase 2
        phase2_config = EasyDict(copy.deepcopy(dict(config)))
        phase2_config.Trainer.max_epochs = phase2_epochs
        
        print(f"\n[Phase 2] Molecule Training: {phase2_epochs} epochs")
        print("  - Freezing: Protein Tower")
        print("  - Training: Molecular Tower + Projection Layers")
        print("-"*80)
        
        model.set_training_phase("molecule_training")
        trainer2 = load_trainer(phase2_config)
        trainer2.fit(model=model, datamodule=data_module)
        
        # Use the last trainer for testing
        trainer = trainer2
    else:
        # Standard training (all components train together)
        print("\n[Standard Training] All components training together")
        trainer = load_trainer(config)
        trainer.fit(model=model, datamodule=data_module)

    # Load best model and test performance
    if model.save_path is not None:
        if config.model.kwargs.get("use_lora", False):
            # Load LoRA model
            config.model.kwargs.lora_config_path = model.save_path
            model = load_model(config.model)

        else:
            model.load_checkpoint(model.save_path, load_prev_scheduler=model.load_prev_scheduler)

    trainer.test(model=model, datamodule=data_module)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
    return parser.parse_args()


def main(args):
    with open(args.config, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))

    if config.setting.seed:
        setup_seed(config.setting.seed)

    if 'os_environ' not in config.setting or config.setting.os_environ is None:
        config.setting.os_environ = EasyDict()

    for k, v in config.setting.os_environ.items():
        if v is not None and k not in os.environ:
            os.environ[k] = str(v)

        elif k in os.environ:
            # override the os environment variables
            config.setting.os_environ[k] = os.environ[k]

    node_rank = int(config.setting.os_environ.get('NODE_RANK', 0))
    
    if node_rank != 0:
        config.Trainer.logger = False

    run(config)


if __name__ == '__main__':
    main(get_args())