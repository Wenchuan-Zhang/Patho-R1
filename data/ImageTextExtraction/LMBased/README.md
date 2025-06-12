## Qwen2.5VL based image caption extraction

### Dependencies

LM-based extraction is based on Qwen API and [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)

```
pip install doclayout-yolo
```

Put DocLayout-YOLO model weights in `data/ImageTextExtraction/models`. Or use the download script

```
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
```

### Default Output Structure

```
output
├── crops/                # Cropped ROIs
│   ├── 0001_fig0.jpg
│   ├── 0001_fig1.jpg
│   └── ...
├── high_res/             # High-res pages
│   ├── 0001.jpg
│   └── ...
├── images/               # Low-res pages with layout annotation
│   ├── 0001.jpg
│   ├── 0001_ali.json
│   └── ...
└── results/              # Image caption pairs
    └── result.json
```

### Processing Pipeline

The book pdf goes through a 5 step pipeline to extract image-caption pairs:

1. **PDF to image conversion.**

   In this step, document pages will be converted into images with sizes optimized for Qwen2.5VL layout analysis, i.e., multiples of `28 * 28` and below `1280 * 28 * 28`. However, although best for layout analysis, such resolution is relatively low for subsequent tasks like image cropping and OCR, where regions of interests may suffer from overly downsampling and lose important visual information. To preserve high-res cropped images as well as fit with layout analysis process, each single page is converted into images of 2 resolution, a lower one for layout analysis, and a higher one (4x the lower one) for cropping ROIs.

2. **Layout analysis**
   
   In this step, page images will be annotated using Qwen2.5VL. The caption region will be transformed into text in this step if using Qwen2.5VL. Additinally, DocLayoutYolo and Qwen-VL-plus are provided as an alternative for perform layout analysis locally and OCR cost-efficiently (only when there is no multi-panel images). 

   Considering processing all pages (both with images and without) is extremely expensive, DocLayoutYolo is used for pre-parsing, detecting whether there are images in the page. Only page meets the following conditions will be processed:

   - Contains image. This page will be processed by Qwen2.5VL for document parsing.

   - The previous page or the following page contains image. This page will be processed by Qwen-max for OCR. (The textual data will be used in the in-line extraction)

   Note that although unnecessary processing is optimized and taken into consideration, this process can still be expensive. Approximately ¥30 will be cost for processing an average book.

3. **Cropping**
   
   Cropping ROIs, i.e., image regions according to the layout annotation of step 2. Note that annotations are based on the low-res images, so cropping positions are calculated accordingly to crop images from high-res document pages to better preserve visual detail information.

4. **In-line reference extraction**
   
   Extract in-line reference through LLM-based OCR. This process has 2 sub-steps: First, use Qwen-plus for OCR of the whole document page to extracrt all body text; then provide the model with image caption text in each page and the body text in the adjacent 3 pages.
   
   Only MLLM based OCR is supported here, as distortions and skew of characters still remain a tricky problem in traditional OCR methods, whereas MLLM-based OCR is robust and cost-efficient. This process cost much less than layout analysis, usually less than ¥3 for an average book (1000 pages).

5. **Processing results**

   In this step, image-caption-inline triplets will be combined into one json file for ease of model training.

### Usage

Sample Usage

```
# bin/bash
export DASHSCOPE_API_KEY=your_api_key_here
cd data/ImageTextExtraction/LMBased
python main.py --pdf ../test.pdf --output ./output --inline True --layout_ocr False --steps yyyyy
```
