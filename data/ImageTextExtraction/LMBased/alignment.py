import cv2
import os
import json
from typing import List, Dict, Tuple, Optional, Any
    
def process_results(self, image_paths: List[str], analysis_results: Dict) -> str:
    """
    process results and crop regions of interest. 
    Only be used in non-LLM based document analysis
    """

    book_results = []
    
    for page_num, analysis in analysis_results.items():

        img_path = None
        for path in image_paths:
            if self._extract_page_number(path) == page_num:
                img_path = path
                break
                
        if not img_path or "figures" not in analysis:
            continue
            
        img = cv2.imread(img_path)
        
        for figure in analysis["figures"]:
            figure_id = figure.get("id", f"fig_{page_num}_{len(book_results)}")
            bbox = figure.get("bbox")
            caption = figure.get("caption", "")
            inline_ref = figure.get("inline_reference", "")
            
            if not bbox or len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cropped_img = img[y1:y2, x1:x2]
            
            crop_filename = f"crop_{page_num}_{figure_id}.jpg"
            crop_path = os.path.join(self.config["crop_dir"], crop_filename)
            cv2.imwrite(crop_path, cropped_img)
            
            book_results.append({
                "page": page_num,
                "figure_id": figure_id,
                "image_path": crop_path,
                "bbox": bbox,
                "caption": caption,
                "inline_reference": inline_ref
            })
    
    result_path = os.path.join(self.config["result_dir"], f"analysis_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(book_results, f, ensure_ascii=False, indent=2)
        
    return result_path


def merge_result(img_dir: str, result_path: str) -> None:
    
    json_files = sorted([f for f in os.listdir(img_dir) if f.endswith('_ali.json')])
    merged = []

    for json_file in json_files:
        json_path = os.path.join(img_dir, json_file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            page_data = json.load(f)
            merged.extend(page_data)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)