#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

try:
    import open_clip
except ImportError:
    print("Error: open_clip is not installed. Please install it using:")
    print("pip install open_clip_torch")
    sys.exit(1)

warnings.filterwarnings('ignore', category=UserWarning)

def setup_logging(log_file=None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

def check_paths(train_root, test_root):
    if not os.path.exists(train_root):
        raise FileNotFoundError(f"Training data path not found: {train_root}")
    if not os.path.exists(test_root):
        raise FileNotFoundError(f"Test data path not found: {test_root}")
    
    logging.info(f"Training data path: {train_root}")
    logging.info(f"Test data path: {test_root}")

def load_clip_model(model_name, pretrained_path, device):
    try:
        logging.info(f"Loading CLIP model: {model_name}")
        logging.info(f"Pretrained weights: {pretrained_path}")
        
        if pretrained_path and not os.path.exists(pretrained_path):
            logging.warning(f"Pretrained path not found: {pretrained_path}")
            logging.info("Using default pretrained weights...")
            pretrained_path = 'openai'
        
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_path
        )
        clip_model = clip_model.to(device)
        clip_model.eval()
        
        logging.info(f"Model loaded successfully on device: {device}")
        return clip_model, preprocess
    
    except Exception as e:
        logging.error(f"Error loading CLIP model: {str(e)}")
        raise

def extract_features(loader, clip_model, device):
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbs in tqdm(loader, desc="Extracting features", leave=False):
            imgs = imgs.to(device)
            try:
                f = clip_model.encode_image(imgs)
                feats.append(f.cpu().numpy())
                labels.append(lbs.numpy())
            except Exception as e:
                logging.error(f"Error during feature extraction: {str(e)}")
                raise
    
    return np.concatenate(feats), np.concatenate(labels)

def sample_few_shot_data(train_dataset, shot, subset_id, global_seed):
    all_targets = np.array(train_dataset.targets)
    num_classes = len(train_dataset.classes)
    
    np.random.seed(global_seed + subset_id)
    train_idx = []
    
    for cls in range(num_classes):
        idxs = np.where(all_targets == cls)[0]
        if len(idxs) == 0:
            logging.warning(f"No samples found for class {cls}")
            continue
        chosen = np.random.choice(idxs, size=min(shot, len(idxs)), replace=False)
        train_idx.extend(chosen.tolist())
    
    logging.info(f"Sampled {len(train_idx)} training samples for {shot}-shot learning")
    return train_idx

def evaluate_few_shot(args):

    log_file = f"few_shot_evaluation_{args.dataset}.log"
    setup_logging(log_file)
    
    logging.info("="*50)
    logging.info("Few-shot Learning Evaluation Started")
    logging.info("="*50)
    
    check_paths(args.train_root, args.test_root)
    
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
    logging.info(f"Using device: {device}")

    clip_model, preprocess = load_clip_model(args.model_name, args.pretrained_path, device)
    
    try:
        train_dataset = ImageFolder(root=args.train_root, transform=preprocess)
        test_dataset = ImageFolder(root=args.test_root, transform=preprocess)
        
        logging.info(f"Train dataset: {len(train_dataset)} samples, {len(train_dataset.classes)} classes")
        logging.info(f"Test dataset: {len(test_dataset)} samples")
        logging.info(f"Classes: {train_dataset.classes}")
        
    except Exception as e:
        logging.error(f"Error loading datasets: {str(e)}")
        raise
    
    all_results = []
    
    for shot in args.shot_list:
        logging.info(f"\n--- Evaluating {shot}-shot learning ---")
        accs = []
        
        for subset_id in range(1, args.n_subsets + 1):
            try:
                train_idx = sample_few_shot_data(train_dataset, shot, subset_id, args.global_seed)
                
                train_loader = DataLoader(
                    Subset(train_dataset, train_idx), 
                    batch_size=args.batch_size, 
                    shuffle=False, 
                    num_workers=args.num_workers
                )
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=False, 
                    num_workers=args.num_workers
                )
                
                logging.info(f"Shot={shot}, subset={subset_id}: extracting features...")
                X_train, y_train = extract_features(train_loader, clip_model, device)
                X_test, y_test = extract_features(test_loader, clip_model, device)
                
                clf = LogisticRegression(
                    random_state=args.global_seed + subset_id,
                    C=args.reg_strength,
                    max_iter=args.max_iter,
                    multi_class='multinomial',
                    solver='lbfgs'
                )
                
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds) * 100
                
                logging.info(f"Shot={shot}, subset={subset_id}: Accuracy={acc:.2f}%")
                accs.append(acc)
                
            except Exception as e:
                logging.error(f"Error in shot={shot}, subset={subset_id}: {str(e)}")
                accs.append(0.0)  
        
        row = {'dataset': args.dataset, 'shot': shot}
        for i, a in enumerate(accs, start=1):
            row[f'acc_subset_{i}'] = a
        
        valid_accs = [a for a in accs if a > 0]
        if valid_accs:
            row['mean_acc'] = np.mean(valid_accs)
            row['std_acc'] = np.std(valid_accs)
            row['min_acc'] = np.min(valid_accs)
            row['max_acc'] = np.max(valid_accs)
            logging.info(f"Shot={shot}: Mean={row['mean_acc']:.2f}±{row['std_acc']:.2f}%")
        
        all_results.append(row)
    
    results_df = pd.DataFrame(all_results)
    out_file = f"{args.output_prefix}_{args.dataset}_summary.xlsx"
    
    try:
        results_df.to_excel(out_file, index=False, float_format='%.4f')
        logging.info(f"Results saved to: {out_file}")
        
        csv_file = out_file.replace('.xlsx', '.csv')
        results_df.to_csv(csv_file, index=False, float_format='%.4f')
        logging.info(f"Results also saved to: {csv_file}")
        
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        csv_file = out_file.replace('.xlsx', '.csv')
        results_df.to_csv(csv_file, index=False, float_format='%.4f')
        logging.info(f"Results saved to CSV: {csv_file}")
    
    logging.info("Few-shot evaluation completed successfully!")
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Few-shot Learning Evaluation using CLIP')
    
    parser.add_argument('--train_root', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--test_root', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--dataset', type=str, default='WSSSLUAD',
                        help='Dataset name for logging')
    
    parser.add_argument('--model_name', type=str, default='ViT-B-16',
                        help='CLIP model name')
    parser.add_argument('--pretrained_path', type=str, default='openai',
                        help='Path to pretrained weights or "openai" for default')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    
    parser.add_argument('--shot_list', type=int, nargs='+', default=[2, 8, 16, 32, 64, 128],
                        help='List of shot numbers to evaluate')
    parser.add_argument('--n_subsets', type=int, default=10,
                        help='Number of random subsets for each shot')
    parser.add_argument('--global_seed', type=int, default=42,
                        help='Global random seed')
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--reg_strength', type=float, default=1.0,
                        help='Regularization strength for logistic regression')
    parser.add_argument('--max_iter', type=int, default=1000,
                        help='Maximum iterations for logistic regression')
    
    parser.add_argument('--output_prefix', type=str, default='few_shot_evaluation',
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_few_shot(args)
        print("\nEvaluation Summary:")
        print(results[['dataset', 'shot', 'mean_acc', 'std_acc']].to_string(index=False))
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()