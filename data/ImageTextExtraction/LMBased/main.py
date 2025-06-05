import os
import sys
import re
import asyncio
from crop import cropping
import prompts
from pdf2img import DocProcessor
from typing import Dict, List, Tuple, Optional, Any
from layout_async import analyze_documents_async
from alignment import merge_result
from inline_async import inline_ref_async
import argparse


def main(pdf_path: str, config: Dict = None) -> str:

    if config is None:
        config = {
            "output_dir": "output",
            "img_dir": "output/images",
            "crop_dir": "output/crops",
            "result_dir": "output/results",
            "qwen_api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "inline": True,
            "layout_ocr": False,
            "steps": "yyyyy",
            "max_retry": 3,
            "retry_interval": 5
        }
    
    # step 1: PDF to images
    if config["steps"][0] == "y":

        processor = DocProcessor(config)
        print("Step 1: Converting PDF to images...")
        processor.pdf_to_optimal_images(pdf_path)
    

    # step 2: layout analysis
    if config["steps"][1] == "y":

        try:
            print("Step 2: Analyzing documents with QwenVL...")
            # analysis_results = await analyze_documents(image_paths)
            asyncio.run(analyze_documents_async(config["qwen_api_url"], config["img_dir"], main_text=config["inline"], use_ocr=config["layout_ocr"]))
            # analyze_documents(config["qwen_api_url"], config["img_dir"])
        except Exception as e:
            print(e)
            print("Error in layout analysis. Skipping...")
            
    # step 3: cropping
    if config["steps"][2] == "y":
        try:
            print("Step 3: Cropping images...")
            cropping(config['img_dir'], config['high_res_dir'], config['crop_dir'])
        except Exception as e:
            print(e)
            print("Error in cropping. Skipping...")


    # step 4: in-line reference extraction
    if config["steps"][3] == "y":
        try:
            print("Step 4: Extracting in-line references...")
            # inline_ref(config['img_dir'], config['qwen_api_url'])
            if config['inline']:
                asyncio.run(inline_ref_async(config['img_dir'], config['qwen_api_url']))
        except Exception as e:
            print(e)
            print("Error in in-line reference extraction. Skipping...")
                
    # step 5: process results
    if config["steps"][4] == "y":
        print("Step 3: Processing results...")
        merge_result(config['img_dir'], os.path.join(config['result_dir'], "result.json"))
    
    print("Processing complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Medical document layout analysis tool")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--api_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                        help="Qwen API URL")
    parser.add_argument("--inline", default=True, help="Extract in-line references")
    parser.add_argument("--layout_ocr", default=False, 
                        help="Use doclayoutYolo for layout analysis, and QwenOCR for main text extraction. Set False to use QwenVL2.5")
    parser.add_argument("--steps", default="yyyyy", help="5 steps of processing. y for yes, n for no. For example, 'yynnn' means only pdf2img and layout analysis will be performed.")

    args = parser.parse_args()
    if args.inline == "False":
        args.inline = False

    if args.layout_ocr == "True":
        args.layout_ocr = True
    
    config = {
        "output_dir": args.output,
        "img_dir": os.path.join(args.output, "images"),
        "high_res_dir": os.path.join(args.output, "high_res"),
        "crop_dir": os.path.join(args.output, "crops"),
        "result_dir": os.path.join(args.output, "results"),
        "inline": args.inline,
        "layout_ocr": args.layout_ocr,
        "qwen_api_url": args.api_url,
        "steps": args.steps,
        "max_retry": 3,
        "retry_interval": 5
    }

    if re.search(r'[\u4e00-\u9fff]', args.pdf.split('/')[-1]):
        prompts.language = "zh"
    else:
        prompts.language = "en"

    print(f"using args: {args}")

    main(args.pdf, config)