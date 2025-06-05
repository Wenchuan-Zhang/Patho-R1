import os
import re
import time
import asyncio
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import cv2
import fitz
from typing import List, Dict, Tuple, Optional, Any


class DocProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.qwen_api_url = config.get("qwen_api_url", "http://localhost:8000/v1/chat/completions")
        self.max_retry = config.get("max_retry", 3)
        self.retry_interval = config.get("retry_interval", 5)
        os.makedirs(config.get("output_dir", "output"), exist_ok=True)
        os.makedirs(config.get("img_dir", "images"), exist_ok=True)
        os.makedirs(config.get("high_res_dir", "high_res"), exist_ok=True)
        os.makedirs(config.get("crop_dir", "crops"), exist_ok=True)
        os.makedirs(config.get("result_dir", "results"), exist_ok=True)

    def pdf_to_optimal_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF to images with optimized size, multiples of 28 * 28 for better tokenizing
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of paths to the converted images
        """
        output_img_dir = self.config["img_dir"]
        high_res_dir = self.config["high_res_dir"]
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(high_res_dir, exist_ok=True)
        
        # Automatically determine optimal image size for document page, i.e. multiples of 28 * 28
        optimal_size = self._analyze_pdf_for_optimal_size(pdf_path)
        print(f"Determined optimal size for images: {optimal_size}")
        high_res_size = (optimal_size[0] * 2, optimal_size[1] * 2)
        force_reprocess = self.config.get("force_reprocess", False) # force process even if images exist
        
        output_paths = self._convert_pdf_to_images_with_resolution(
            pdf_path, output_img_dir, optimal_size, force_reprocess)
        self._convert_pdf_to_images_with_resolution(
            pdf_path, high_res_dir, high_res_size, force_reprocess)

        return output_paths

    def _analyze_pdf_for_optimal_size(self, pdf_path: str) -> Tuple[int, int]:
        """
        Determine optimal image size for PDF documents. To better fit with Qwen2.5 VL, resolutions below 1280*28*28 are recommended.
        """
        pdf_document = fitz.open(pdf_path)
        sample_page_num = min(49, len(pdf_document) - 1)      
        sample_page = pdf_document[sample_page_num]        
        width, height = sample_page.rect.width, sample_page.rect.height        
        aspect_ratio = width / height
        
        max_pixels = 1280 * 28 * 28
        
        # horizontally extended page
        if aspect_ratio > 1:
            w_new = int(np.sqrt(max_pixels * aspect_ratio))
            h_new = int(w_new / aspect_ratio)
        else:  # vertically extended page
            h_new = int(np.sqrt(max_pixels / aspect_ratio))
            w_new = int(h_new * aspect_ratio)
        
        w_new = (w_new // 28) * 28
        h_new = (h_new // 28) * 28        
        w_new = max(28, w_new)
        h_new = max(28, h_new)
        
        pdf_document.close()
        
        return (w_new, h_new)
    
    def _convert_pdf_to_images_mt(self, pdf_path: str, output_dir: str, 
                                size: Tuple[int, int], pages_to_process: List[int] = None) -> List[str]:
        """
        Multithreaded PDF to images conversion with specified size
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save the images
            size: Target image size (width, height)
            pages_to_process: List of page numbers to process (1-indexed)
        Returns:
            List of paths to the generated images
        """
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        if pages_to_process is None:
            pages_to_process = list(range(1, total_pages + 1))
        
        page_indices = [page - 1 for page in pages_to_process if 1 <= page <= total_pages]
        output_paths = []
        
        def process_page(page_num):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))            
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)            
            img_resized = img.resize(size, Image.LANCZOS)            
            output_path = os.path.join(output_dir, f"{page_num+1:04d}.jpg")
            img_resized.save(output_path, quality=95)
            
            return output_path
        
        max_workers = 32
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {executor.submit(process_page, i): i for i in page_indices}            
            for future in tqdm(concurrent.futures.as_completed(future_to_page), 
                              total=len(page_indices), desc=f"Converting {len(page_indices)} PDF pages"):
                page_num = future_to_page[future]
                try:
                    output_path = future.result()
                    output_paths.append(output_path)
                except Exception as e:
                    print(f"Error processing page {page_num+1}: {e}")
        
        pdf_document.close()
        
        # sort with page number
        output_paths.sort()
        
        return output_paths
    
    def _extract_page_number(self, img_path: str) -> str:
        """Extract page number from image filename. Supported formats: page_1.jpg, 1.jpg, etc."""
        filename = os.path.basename(img_path)
        match = re.search(r'(\d+)', filename)
        if match:
            return match.group(1)
        return os.path.splitext(filename)[0]
    
    def _convert_pdf_to_images_with_resolution(
        self, pdf_path: str, output_dir: str, size: Tuple[int, int], 
        force_reprocess: bool = False) -> List[str]:
        """
        Process PDF to images with specific resolution. Breakpoint resume supported
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save the converted images
            size: Target image size (width, height)
            force_reprocess: Whether to reprocess existing images
            
        Returns:
            List of paths to the generated images
        """
        # Get total pages in PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        pdf_document.close()        
        existing_images = sorted([f for f in os.listdir(output_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        output_paths = []
        
        if force_reprocess or len(existing_images) < total_pages:
            print(f"Processing images at {size} resolution: {len(existing_images)}/{total_pages} completed")
            
            if not force_reprocess:
                existing_pages = set()
                for img_name in existing_images:
                    match = re.search(r'(\d+)\.', img_name)
                    if match:
                        existing_pages.add(int(match.group(1)))
                
                pages_to_process = [i for i in range(1, total_pages + 1) if i not in existing_pages]
            else:
                pages_to_process = list(range(1, total_pages + 1))
            
            if pages_to_process:

                new_paths = self._convert_pdf_to_images_mt(
                    pdf_path, output_dir, size, pages_to_process)
                output_paths.extend(new_paths)
            
            # Merge with existing results if not reprocessing everything
            if not force_reprocess:
                existing_paths = [os.path.join(output_dir, img) for img in existing_images]
                output_paths.extend(existing_paths)
                output_paths.sort()
        else:
            output_paths = [os.path.join(output_dir, img) for img in existing_images]
            
        return output_paths

