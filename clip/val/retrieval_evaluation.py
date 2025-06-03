#!/usr/bin/env python3

import os
import json
import torch
import argparse
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip
import gc
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP model on text-to-image retrieval task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-name', 
        type=str, 
        default='ViT-L-14-336',
        help='CLIP model architecture name'
    )
    parser.add_argument(
        '--pretrained-path',
        type=str,
        required=True,
        help='Path to pretrained model checkpoint'
    )
    
    parser.add_argument(
        '--captions-json',
        type=str,
        required=True,
        help='Path to captions JSON file'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        required=True,
        help='Directory containing images'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for image encoding'
    )
    parser.add_argument(
        '--text-batch-size',
        type=int,
        default=256,
        help='Batch size for text encoding'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for computation'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of workers for data loading'
    )
    
    parser.add_argument(
        '--recall-k',
        type=int,
        nargs='+',
        default=[1, 5, 10, 20],
        help='K values for Recall@K evaluation'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (for testing)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--save-features',
        action='store_true',
        help='Save encoded features to disk'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def setup_device(device_arg):
    if device_arg == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def load_and_validate_data(captions_json, images_dir, max_samples=None, verbose=False):
    print("Loading captions and validating data...")
    
    captions_path = Path(captions_json)
    images_path = Path(images_dir)
    
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")
    
    with open(captions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Sort by numeric key for consistent ordering
    uuids = []
    captions = []
    for key, entry in sorted(data.items(), key=lambda x: int(x[0])):
        if 'uuid' not in entry or 'caption' not in entry:
            if verbose:
                print(f"Warning: skipping malformed entry {key}")
            continue
        uuids.append(entry['uuid'])
        captions.append(entry['caption'])
    
    # Build mapping from UUID stem to image path
    stem_to_path = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    for p in images_path.iterdir():
        if p.is_file() and p.suffix.lower() in image_extensions:
            stem_to_path[p.stem] = p
    
    # Filter out entries without images and validate
    valid_uuids = []
    valid_captions = []
    missing_images = []
    
    for uid, cap in zip(uuids, captions):
        if uid in stem_to_path:
            try:
                with Image.open(stem_to_path[uid]) as img:
                    img.verify()  
                valid_uuids.append(uid)
                valid_captions.append(cap)
            except Exception as e:
                if verbose:
                    print(f"Warning: corrupted image for UUID {uid}: {e}")
                missing_images.append(uid)
        else:
            missing_images.append(uid)
        
        if max_samples and len(valid_uuids) >= max_samples:
            break
    
    print(f"Total entries: {len(uuids)}")
    print(f"Valid entries: {len(valid_uuids)}")
    print(f"Missing/corrupted images: {len(missing_images)}")
    
    if len(valid_uuids) == 0:
        raise ValueError("No valid image-caption pairs found!")
    
    return valid_uuids, valid_captions, stem_to_path


def encode_texts_in_batches(model, tokenizer, captions, device, batch_size=256, verbose=False):
    print(f"Encoding {len(captions)} text captions in batches...")
    text_features = []
    
    desc = "Encoding texts" if not verbose else "Encoding texts (verbose)"
    for start in tqdm(range(0, len(captions), batch_size), desc=desc):
        batch_captions = captions[start:start + batch_size]
        with torch.no_grad():
            text_tokens = tokenizer(batch_captions).to(device)
            feats = model.encode_text(text_tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            text_features.append(feats.cpu())  
            
        del text_tokens, feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return torch.cat(text_features, dim=0)


def encode_images_in_batches(model, preprocess, uuids, stem_to_path, device, batch_size=64, verbose=False):
    print(f"Encoding {len(uuids)} images in batches...")
    image_features = []
    error_count = 0
    
    desc = "Encoding images" if not verbose else "Encoding images (verbose)"
    for start in tqdm(range(0, len(uuids), batch_size), desc=desc):
        batch_uuids = uuids[start:start + batch_size]
        imgs = []
        
        for uid in batch_uuids:
            try:
                img_path = stem_to_path[uid]
                img = Image.open(img_path).convert('RGB')
                imgs.append(preprocess(img))
            except Exception as e:
                if verbose:
                    print(f"Error loading image {uid}: {e}")
                error_count += 1
                imgs.append(torch.zeros_like(preprocess(Image.new('RGB', (224, 224)))))
        
        if imgs:
            batch = torch.stack(imgs, dim=0).to(device)
            with torch.no_grad():
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                image_features.append(feats.cpu())  
            
            del batch, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    if error_count > 0:
        print(f"Warning: {error_count} images failed to load")
    
    return torch.cat(image_features, dim=0)


def compute_retrieval_metrics(text_feats, image_feats, k_values=[1, 5, 10, 20], device=None):
    print("Computing similarity and evaluating retrieval metrics...")
    
    N = text_feats.size(0)
    assert N == image_feats.size(0), "Text and image features must have same number of samples"
    
    if device and device.type == 'cuda' and N < 10000:
        text_feats = text_feats.to(device)
        image_feats = image_feats.to(device)
        sim_matrix = image_feats @ text_feats.T
        device_compute = True
    else:
        device_compute = False
    
    recall_counters = {k: 0 for k in k_values}
    max_k = max(k_values)
    
    if device_compute:
        
        for i in tqdm(range(N), desc="Computing recall"):
            topk_indices = sim_matrix[i].topk(max_k, largest=True).indices
            for k in k_values:
                if i in topk_indices[:k]:
                    recall_counters[k] += 1
    else:
        
        batch_size = min(100, N // 10 + 1)  
        for start in tqdm(range(0, N, batch_size), desc="Computing recall"):
            end = min(start + batch_size, N)
            batch_text_feats = text_feats[start:end]
            
            
            sim_batch = batch_text_feats @ image_feats.T
            
            for i, global_i in enumerate(range(start, end)):
                topk_indices = sim_batch[i].topk(max_k, largest=True).indices
                for k in k_values:
                    if global_i in topk_indices[:k]:
                        recall_counters[k] += 1
    
   
    recall_metrics = {k: (count / N) * 100 for k, count in recall_counters.items()}
    
    return recall_metrics, N


def save_results(results, args, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"clip_evaluation_{timestamp}.json"
    
    save_data = {
        'timestamp': timestamp,
        'args': vars(args),
        'results': results,
        'recall_metrics': results['recall_metrics'],
        'num_samples': results['num_samples']
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {results_file}")
    return results_file


def main():
    args = parse_args()
    
    device = setup_device(args.device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 1. Load model and transforms
        print("Loading model...")
        print(f"Model: {args.model_name}")
        print(f"Checkpoint: {args.pretrained_path}")
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model_name, pretrained=args.pretrained_path)
        tokenizer = open_clip.get_tokenizer(args.model_name)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
        
        # 2. Load and validate data
        valid_uuids, valid_captions, stem_to_path = load_and_validate_data(
            args.captions_json, args.images_dir, args.max_samples, args.verbose)
        
        # 3. Encode features
        text_feats = encode_texts_in_batches(
            model, tokenizer, valid_captions, device, 
            batch_size=args.text_batch_size, verbose=args.verbose)
        
        image_feats = encode_images_in_batches(
            model, preprocess, valid_uuids, stem_to_path, device, 
            batch_size=args.batch_size, verbose=args.verbose)
        
        # 4. Save features if requested
        if args.save_features:
            features_dir = Path(args.output_dir) / "features"
            features_dir.mkdir(exist_ok=True)
            
            torch.save(text_feats, features_dir / "text_features.pt")
            torch.save(image_feats, features_dir / "image_features.pt")
            print(f"Features saved to: {features_dir}")
        
        # 5. Compute retrieval metrics
        recall_metrics, N = compute_retrieval_metrics(
            text_feats, image_feats, args.recall_k, device)
        
        # 6. Print and save results
        print(f"\n{'='*60}")
        print(f"CLIP Model Text-to-Image Retrieval Evaluation Results")
        print(f"{'='*60}")
        print(f"Model: {args.model_name}")
        print(f"Checkpoint: {Path(args.pretrained_path).name}")
        print(f"Samples: {N}")
        print(f"{'='*60}")
        for k in sorted(recall_metrics.keys()):
            print(f"Recall@{k:2d}: {recall_metrics[k]:6.2f}%")
        print(f"{'='*60}")
        
        results = {
            'recall_metrics': recall_metrics,
            'num_samples': N,
            'model_name': args.model_name,
            'checkpoint_path': args.pretrained_path
        }
        
        results_file = save_results(results, args, args.output_dir)
        
        del text_feats, image_feats, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Evaluation completed successfully!")
        print(f"Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()