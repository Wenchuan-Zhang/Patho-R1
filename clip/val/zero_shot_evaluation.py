#!/usr/bin/env python3
"""
Zero-shot Classification Evaluation for Medical Image Datasets using CLIP
"""

import os
import json
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import torch
import open_clip
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def setup_logging(log_dir="./logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"zero_shot_eval_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_file):
    """Load dataset configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config file {config_file}: {e}")
        return None

def validate_paths(datasets):
    """Validate that all dataset paths exist"""
    for ds in datasets:
        if not os.path.exists(ds['prompt']):
            logging.error(f"Prompt file not found: {ds['prompt']}")
            return False
        if not os.path.exists(ds['data_dir']):
            logging.error(f"Data directory not found: {ds['data_dir']}")
            return False
    return True

def load_model(model_name, pretrained_path, device):
    """Load OpenCLIP model and transforms"""
    try:
        logging.info(f"Loading model: {model_name}")
        logging.info(f"Pretrained path: {pretrained_path}")
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_path
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model = model.to(device)
        model.eval()
        
        logging.info("Model loaded successfully")
        return model, preprocess, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None, None, None

def build_zeroshot_weights(model, tokenizer, class_texts, templates, device):
    """Build zero-shot classification weights"""
    logging.info("Building zero-shot weights...")
    zeroshot_weights = []
    
    with torch.no_grad():
        for i, names in enumerate(class_texts):
            embeds = []
            for nm in names:
                texts = [t.replace('CLASSNAME', nm) for t in templates]
                tokens = tokenizer(texts).to(device)
                text_feats = model.encode_text(tokens)
                text_feats /= text_feats.norm(dim=-1, keepdim=True)
                embeds.append(text_feats)
            
            embeds = torch.cat(embeds, dim=0)
            cls_feat = embeds.mean(dim=0)
            cls_feat /= cls_feat.norm()
            zeroshot_weights.append(cls_feat)
            
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    logging.info(f"Zero-shot weights shape: {zeroshot_weights.shape}")
    return zeroshot_weights

def evaluate_dataset(model, loader, zeroshot_weights, device, dataset_name):
    """Evaluate model on a single dataset"""
    logging.info(f"Evaluating {dataset_name}...")
    
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Processing {dataset_name}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Encode images
            img_feats = model.encode_image(imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            
            # Calculate logits
            scale = model.logit_scale.exp().item() if hasattr(model, 'logit_scale') else 100.0
            logits = scale * img_feats @ zeroshot_weights
            preds = logits.argmax(dim=1)
            
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())
    
    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    
    # Calculate metrics
    acc = accuracy_score(labels_all.numpy(), preds_all.numpy())
    
    return acc, preds_all.numpy(), labels_all.numpy()

def save_results(results, output_dir="./results", detailed_results=None):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary results
    summary_file = os.path.join(output_dir, f"zero_shot_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed results if provided
    if detailed_results:
        detailed_file = os.path.join(output_dir, f"zero_shot_detailed_{timestamp}.json")
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
    
    logging.info(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation using CLIP")
    parser.add_argument("--config", type=str, help="Dataset configuration file")
    parser.add_argument("--model_name", type=str, default="ViT-B-16", 
                       help="OpenCLIP model name")
    parser.add_argument("--pretrained_path", type=str, required=True,
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Log directory")
    parser.add_argument("--detailed_metrics", action="store_true",
                       help="Compute detailed metrics (classification report, confusion matrix)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting zero-shot evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Default datasets if no config provided
    if args.config:
        config = load_config(args.config)
        if config is None:
            return 1
        datasets = config.get('datasets', [])
    else:
        datasets = [
            {
                "name": "LC-Lung",
                "prompt": "/path/to/lung_prompt.json",
                "data_dir": "/path/to/lung_image_sets"
            }
        ]
    
    # Validate paths
    if not validate_paths(datasets):
        logger.error("Path validation failed")
        return 1
    
    # Load model
    model, preprocess, tokenizer = load_model(args.model_name, args.pretrained_path, device)
    if model is None:
        return 1
    
    # Main evaluation loop
    results = {}
    detailed_results = {}
    start_time = time.time()
    
    for ds in datasets:
        dataset_start_time = time.time()
        name = ds['name']
        prompt_file = ds['prompt']
        data_dir = ds['data_dir']
        
        logger.info(f"\n=== Evaluating {name} ===")
        
        try:
            # Load dataset
            dataset = ImageFolder(data_dir, transform=preprocess)
            loader = DataLoader(dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=args.num_workers)
            idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
            logger.info(f"Samples: {len(dataset)}, Classes: {len(idx_to_class)}")
            
            # Load prompt JSON
            with open(prompt_file, 'r') as f:
                prompt_data = json.load(f)['0']
            classnames_dict = prompt_data['classnames']
            templates = prompt_data['templates']
            
            # Gather names per class
            class_texts = [classnames_dict[idx_to_class[i]] for i in range(len(dataset.classes))]
            for i, names in enumerate(class_texts):
                logger.info(f"Class {i} ({idx_to_class[i]}): {', '.join(names)}")
            
            # Build zero-shot weights
            zeroshot_weights = build_zeroshot_weights(
                model, tokenizer, class_texts, templates, device
            )
            
            # Evaluate
            acc, preds, labels = evaluate_dataset(
                model, loader, zeroshot_weights, device, name
            )
            
            results[name] = acc
            dataset_time = time.time() - dataset_start_time
            logger.info(f"{name} Accuracy: {acc:.4f} (Time: {dataset_time:.2f}s)")
            
            # Detailed metrics if requested
            if args.detailed_metrics:
                class_names = [idx_to_class[i] for i in range(len(dataset.classes))]
                report = classification_report(labels, preds, target_names=class_names, output_dict=True)
                cm = confusion_matrix(labels, preds)
                
                detailed_results[name] = {
                    'accuracy': acc,
                    'classification_report': report,
                    'confusion_matrix': cm.tolist(),
                    'class_names': class_names,
                    'evaluation_time': dataset_time
                }
            
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            results[name] = None
    
    total_time = time.time() - start_time
    logger.info(f"\nTotal evaluation time: {total_time:.2f}s")
    
    # Print summary
    logger.info("\n=== Zero-Shot Accuracy Summary ===")
    for k, v in results.items():
        if v is not None:
            logger.info(f"{k}: {v:.4f}")
        else:
            logger.info(f"{k}: FAILED")
    
    # Calculate average accuracy (excluding failed evaluations)
    valid_results = [v for v in results.values() if v is not None]
    if valid_results:
        avg_acc = np.mean(valid_results)
        logger.info(f"Average Accuracy: {avg_acc:.4f}")
        results['average'] = avg_acc
    
    # Save results
    save_results(results, args.output_dir, detailed_results if args.detailed_metrics else None)
    
    logger.info("Evaluation completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())