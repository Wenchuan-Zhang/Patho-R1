#!/usr/bin/env python3
"""
WebDataset Creator - Convert CSV with image paths to WebDataset tar shards

This script processes a CSV file containing image paths and captions,
creating tar files suitable for WebDataset format with parallel processing.
"""

import os
import io
import sys
import math
import json
import tarfile
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_shard(shard_idx: int, df_chunk: pd.DataFrame, image_dir: str, out_dir: str) -> int:
    """
    Process one DataFrame chunk into a single .tar shard.
    
    Args:
        shard_idx: Index of the shard
        df_chunk: DataFrame chunk to process
        image_dir: Directory containing images
        out_dir: Output directory for tar files
        
    Returns:
        shard_idx: The processed shard index
    """
    tar_name = f"{shard_idx:05d}.tar"
    tar_path = os.path.join(out_dir, tar_name)
    
    processed_count = 0
    error_count = 0
    
    try:
        with tarfile.open(tar_path, 'w') as tar:
            for inner_idx, row in df_chunk.iterrows():
                key = f"{shard_idx:05d}{inner_idx % len(df_chunk):04d}"
                img_path = os.path.join(image_dir, row['image_path'])
                caption = str(row['caption'])  # Ensure caption is string
                
                # Check if image file exists
                if not os.path.exists(img_path):
                    tqdm.write(f"[Shard {shard_idx}] ⚠️  Image not found: {img_path}")
                    error_count += 1
                    continue
                
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                        # Convert to RGB if needed (for consistency)
                        if im.mode not in ['RGB', 'L']:
                            im = im.convert('RGB')
                except Exception as e:
                    tqdm.write(f"[Shard {shard_idx}] ⚠️  Failed to open {img_path}: {e}")
                    error_count += 1
                    continue

                # 1) Add image file
                ext = os.path.splitext(row['image_path'])[1].lower()
                if not ext:
                    ext = '.jpg'  # Default extension
                img_name = key + ext
                tar.add(img_path, arcname=img_name)

                # 2) Add JSON metadata
                meta = {
                    "caption": caption,
                    "key": key,
                    "width": w,
                    "height": h,
                    "original_path": row['image_path']
                }
                js = json.dumps(meta, ensure_ascii=False, indent=None).encode('utf-8')
                ti = tarfile.TarInfo(name=key + ".json")
                ti.size = len(js)
                tar.addfile(ti, io.BytesIO(js))

                # 3) Add plain-text caption
                txt = caption.encode('utf-8')
                ti = tarfile.TarInfo(name=key + ".txt")
                ti.size = len(txt)
                tar.addfile(ti, io.BytesIO(txt))
                
                processed_count += 1

    except Exception as e:
        logger.error(f"Failed to create shard {shard_idx}: {e}")
        raise
    
    if processed_count > 0:
        tqdm.write(f"[Shard {shard_idx}] ✅ Processed {processed_count} samples, {error_count} errors")
    else:
        tqdm.write(f"[Shard {shard_idx}] ❌ No samples processed, {error_count} errors")
    
    return shard_idx


def validate_inputs(csv_path: str, image_dir: str, out_dir: str) -> None:
    """Validate input parameters."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Check CSV format
    try:
        df_sample = pd.read_csv(csv_path, nrows=5)
        required_columns = ['image_path', 'caption']
        for col in required_columns:
            if col not in df_sample.columns:
                raise ValueError(f"Required column '{col}' not found in CSV. Available columns: {list(df_sample.columns)}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")


def create_webdataset(
    csv_path: str,
    image_dir: str,
    out_dir: str,
    samples_per_shard: int = 5000,
    num_workers: Optional[int] = None,
    start_shard_idx: int = 0
) -> None:
    """
    Splits CSV into tar shards, naming shards starting from 'start_shard_idx'.
    
    Args:
        csv_path: Path to CSV file with image_path and caption columns
        image_dir: Directory containing the images
        out_dir: Output directory for tar shards
        samples_per_shard: Number of samples per shard
        num_workers: Number of parallel workers (None for auto)
        start_shard_idx: Starting index for shard naming
    """
    # Validate inputs
    validate_inputs(csv_path, image_dir, out_dir)
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Count rows to estimate number of shards
    logger.info("Counting total samples...")
    try:
        total = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1  # -1 for header
    except Exception as e:
        logger.error(f"Failed to count rows in CSV: {e}")
        raise
    
    n_shards = math.ceil(total / samples_per_shard)
    logger.info(f"Total samples = {total:,}, Shards = {n_shards} (starting at {start_shard_idx})")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Samples per shard: {samples_per_shard}")
    logger.info(f"Workers: {num_workers or 'auto'}")

    # Stream CSV in chunks of size samples_per_shard
    try:
        reader = pd.read_csv(csv_path, encoding='utf-8', chunksize=samples_per_shard)
    except Exception as e:
        logger.error(f"Failed to create CSV reader: {e}")
        raise

    # Launch pool of workers
    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        logger.info("Starting parallel processing...")
        
        for offset, df_chunk in enumerate(reader):
            shard_idx = start_shard_idx + offset
            future = executor.submit(process_shard, shard_idx, df_chunk, image_dir, out_dir)
            futures.append(future)

        # Track progress with tqdm
        completed_shards = 0
        failed_shards = 0
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing shards"):
            try:
                shard_idx = future.result()  # This will re-raise any exception
                completed_shards += 1
            except Exception as e:
                failed_shards += 1
                logger.error(f"Shard processing failed: {e}")
                continue

    logger.info(f"Processing completed: {completed_shards} shards successful, {failed_shards} failed")
    
    if failed_shards > 0:
        logger.warning(f"⚠️  {failed_shards} shards failed to process")
        sys.exit(1)
    else:
        logger.info("✅ All shards processed successfully!")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Create WebDataset tar shards from CSV and images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="Path to CSV file with 'image_path' and 'caption' columns"
    )
    
    parser.add_argument(
        "--images", "-i",
        required=True,
        help="Directory containing the images"
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for tar shards"
    )
    
    parser.add_argument(
        "--samples-per-shard", "-s",
        type=int,
        default=5000,
        help="Number of samples per shard"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto)"
    )
    
    parser.add_argument(
        "--start-shard-idx",
        type=int,
        default=0,
        help="Starting index for shard naming"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Convert paths to absolute paths
    csv_path = os.path.abspath(args.csv)
    image_dir = os.path.abspath(args.images)
    out_dir = os.path.abspath(args.output)
    
    logger.info("Starting WebDataset creation...")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Images: {image_dir}")
    logger.info(f"Output: {out_dir}")
    
    try:
        create_webdataset(
            csv_path=csv_path,
            image_dir=image_dir,
            out_dir=out_dir,
            samples_per_shard=args.samples_per_shard,
            num_workers=args.workers,
            start_shard_idx=args.start_shard_idx
        )
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()