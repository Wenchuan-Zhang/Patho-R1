import asyncio
import os
import re
import json
import base64
import prompts
import time
from tools import get_img_with_ali, get_page_with_context
from typing import List, Dict, Optional, Any
from openai import OpenAI, AsyncOpenAI
from doclayout_yolo import YOLOv10
from tqdm.asyncio import tqdm as async_tqdm
import functools
from crop import cropping_main_text


def update_prompt():
    global system_prompt_alignment, prompt_alignment, system_prompt_recognition, prompt_recognition
    if prompts.language == "zh":
        system_prompt_alignment = prompts.system_prompt_alignment_zh
        prompt_alignment = prompts.prompt_alignment_zh
        system_prompt_recognition = prompts.system_prompt_recognition_zh
        prompt_recognition = prompts.prompt_recognition_zh
    else: 
        system_prompt_alignment = prompts.system_prompt_alignment_en
        prompt_alignment = prompts.prompt_alignment_en
        system_prompt_recognition = prompts.system_prompt_recognition_en
        prompt_recognition = prompts.prompt_recognition_en

def encode_image_to_base64(image_content):
    return base64.b64encode(image_content).decode('utf-8')

async def retry_async(func, max_attempts=3, retry_interval=5):
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            print(f"API request failed (attempt {attempt+1}/{max_attempts}): {str(e)}")
            if attempt == max_attempts - 1:
                return None
            await asyncio.sleep(retry_interval)
    return None

async def send_api_request_async(payload, api_url):
    async def _request():
        async_client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=api_url
        )
        response = await async_client.chat.completions.create(**payload)
        if hasattr(response, "choices") and len(response.choices) > 0:
            return response.choices[0].message.content
        return None
    
    return await retry_async(_request)


async def align_fig_caps_async(img_path: str, api_url: str):
    
    system_prompt = system_prompt_alignment
    user_prompt = prompt_alignment
    
    with open(img_path, "rb") as f:
        img_content = f.read()
        img_base64 = encode_image_to_base64(img_content)

    payload = {
        "model": "qwen2.5-vl-72b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]}
        ],
        "max_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.8,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }
    await asyncio.sleep(1)
    content = await send_api_request_async(payload, api_url)
    if content:
        try:
            if content == "None":
                with open(img_path.replace('.jpg', '_none.json'), 'w') as f:
                    json.dump([], f, ensure_ascii=False)
                return None
            
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content

            if json_str == '[]' or json_str == '' or json_str == 'None' or json_str == 'none' or json_str == 'NONE' or json_str == '[[]]':
                with open(img_path.replace('.jpg', '_none.json'), 'w') as f:
                    json.dump([], f, ensure_ascii=False)
                return None
            
            analysis = json.loads(json_str)
            with open(img_path.replace('.jpg', '_ali.json'), 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=4, ensure_ascii=False)
        
        except Exception as e:
            print(e)
            with open(os.path.join(os.path.dirname(img_path), 'error.txt'), 'a', encoding='utf-8') as f:
                f.write(f"{img_path}\n")

    return None

async def main_text_recognition_async(img_path: str, api_url: str):

    system_prompt = system_prompt_recognition
    user_prompt = prompt_recognition
    
    with open(img_path, "rb") as f:
        img_content = f.read()
        img_base64 = encode_image_to_base64(img_content)
    
    payload = {
        "model": "qwen2.5-vl-72b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]}
        ],
        "max_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.8,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }

    await asyncio.sleep(1)
    content = await send_api_request_async(payload, api_url)
    if content:
        if content == "None":
            with open(img_path.replace('.jpg', '_none.txt'), 'w', encoding='utf-8') as f:
                f.write(content)
            return None

        with open(img_path.replace('.jpg', '_main.txt'), 'w', encoding='utf-8') as f:
            content = re.sub(r"^```markdown\s*|\s*```$", "", content)
            f.write(content)

    return None


async def main_text_recognition_ocr_async(img_path: str, api_url: str):
    
    with open(img_path, "rb") as f:
        img_content = f.read()
        img_base64 = encode_image_to_base64(img_content)

    payload = {
        "model": "qwen-vl-ocr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        "min_pixels": 28 * 28 * 4,
                        "max_pixels": 28 * 28 * 1280
                    },
                    {"type": "text", "text": "Read all the text in the image."},
                ]
            }
        ]
    }

    await asyncio.sleep(1)
    content = await send_api_request_async(payload, api_url)
    if content:
        if content == "None":
            with open(img_path.replace('.jpg', '_none.txt'), 'w', encoding='utf-8') as f:
                f.write(content)
            return None

        with open(img_path.replace('.jpg', '.txt'), 'w', encoding='utf-8') as f:
            # content = re.sub(r"^```markdown\s*|\s*```$", "", content)  # 
            f.write(content)

    return None

async def analyze_documents_async(api_url: str, image_dir: str, concurrency: int = 5, main_text: bool = True, use_ocr: bool = False):

    update_prompt()
    done = 0
    while(done == 0):
        try:
            done = yolo_layout(image_dir, use_ocr)
        except Exception as e:
            print(e)
            time.sleep(3)

    img_files = os.listdir(image_dir)
    img_paths = [os.path.join(image_dir, img) for img in img_files if img.endswith('.jpg')]


    # img_ali_paths = get_img_with_ali(image_dir, absolute_path=True)
    
    img_ali_paths = sorted([img for img in img_paths 
                     if not (os.path.exists(img.replace('.jpg', '_ali.json')) 
                             or os.path.exists(img.replace('.jpg', '_none.json')))])
    
    # img_ali_paths = []

    to_be_processed_img_main_paths = sorted([img for img in img_paths
                      if not (os.path.exists(img.replace('.jpg', '_main.txt'))
                              or os.path.exists(img.replace('.jpg', '_none.txt')))])
    
    # alignment task
    if img_ali_paths:
        print(f"Figure-caption alignment: {len(img_ali_paths)} files")
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(img_file):
            async with semaphore:
                # img_path = os.path.join(image_dir, img_file)
                return await align_fig_caps_async(img_file, api_url)
        
        tasks = [process_with_semaphore(img_file) for img_file in sorted(img_ali_paths)]
        await async_tqdm.gather(*tasks, desc="alignment")
    
    all_img_main_paths = get_page_with_context(image_dir, absolute_path=True)
    img_main_paths = [img for img in all_img_main_paths if img in to_be_processed_img_main_paths]
    # recognition task
    if main_text:
        if not use_ocr:
            if img_main_paths:
                print(f"Main text recognition: {len(img_main_paths)} files")
                semaphore = asyncio.Semaphore(8)
                
                async def process_with_semaphore(img_file):
                    async with semaphore:
                        # img_path = os.path.join(image_dir, img_file)
                        return await main_text_recognition_async(img_file, api_url)
                
                tasks = [process_with_semaphore(img_file) for img_file in sorted(img_main_paths)]
                await async_tqdm.gather(*tasks, desc="recognition")
        else:
            if img_main_paths:
                print(f"Main text recognition: {len(img_main_paths)} files. Using doclayoutYolo & QwenOCR for extraction.")
                layout_path = os.path.join(os.path.dirname(image_dir), 'layout')
                cropping_main_text(image_dir, layout_path)
                to_be_ocr = []
                for __img in img_main_paths:
                    __img_index = __img.split('/')[-1].replace('.jpg', '')
                    __text_folder = os.path.join(layout_path, __img_index)
                    for text_img in os.listdir(__text_folder):
                        if text_img.endswith('.jpg'):
                            to_be_ocr.append(os.path.join(__text_folder, text_img))

                if to_be_ocr:
                    semaphore = asyncio.Semaphore(10)
                    async def process_with_semaphore(img_file):
                        async with semaphore:
                            return await main_text_recognition_ocr_async(img_file, api_url)
                        
                    tasks = [process_with_semaphore(img_file) for img_file in sorted(to_be_ocr)]
                    await async_tqdm.gather(*tasks, desc="recognition (OCR)")

                img_index_folder = [f.split('/')[-1].replace('.jpg', '') for f in img_main_paths]
                for index in img_index_folder:
                    __image_path = os.path.join(image_dir, f"{index}.jpg")
                    __text_folder = os.path.join(layout_path, index)
                    for text_file in os.listdir(__text_folder):
                        if text_file.endswith('.txt'):
                            with open(os.path.join(__text_folder, text_file), 'r', encoding='utf-8') as f:
                                content = f.read()
                            with open(__image_path.replace('.jpg', '_main.txt'), 'a', encoding='utf-8') as f:
                                f.write(content + '\n')

def yolo_layout(img_dir: str, use_ocr: bool = False):
    imgs = os.listdir(img_dir)
    if not os.path.exists(os.path.join(img_dir, 'done.txt')):
        with open(os.path.join(img_dir, 'done.txt'), 'w', encoding='utf-8') as f:
            f.write('')
    with open(os.path.join(img_dir, 'done.txt'), 'r', encoding='utf-8') as f:
        done = f.read().split('\n')
    img_paths = [os.path.join(img_dir, img) for img in imgs if img.endswith('.jpg') and img not in done]
    if use_ocr:
        layout_dir = os.path.join(os.path.dirname(img_dir), 'layout')
        os.makedirs(layout_dir, exist_ok=True)

    model = YOLOv10('../models/doclayout.pt') # path to layout model
    for img_path in img_paths:
        current_image = img_path.split('/')[-1]
        try:
            det_res = model.predict(
                img_path,
                imgsz=1024,
                conf=0.23,
                device='cuda',
            )[0]
            json_content = det_res.tojson()
            if use_ocr:
                with open(os.path.join(layout_dir, current_image.replace('.jpg', '.json')), 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, ensure_ascii=False)

            json_content = json.loads(json_content)

            skip = True
            for item in json_content:
                if item['class'] == 3:
                    skip = False
                    with open(os.path.join(img_dir, 'done.txt'), 'a', encoding='utf-8') as f:
                        f.write(img_path.split('/')[-1] + '\n')
                    break

            if skip:
                with open(img_path.replace('.jpg', '_none.json'), 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)

                with open(os.path.join(img_dir, 'done.txt'), 'a', encoding='utf-8') as f:
                    f.write(img_path.split('/')[-1] + '\n')

        except Exception as e:
            with open(img_path.replace('.jpg', '_none.json'), 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False)

    # if use_ocr:
    #     to_be_ocr = cropping_main_text(img_dir, layout_dir)
    #     for img_path in to_be_ocr:
    #         text_path = img_path.replace('.jpg', '_main.txt')

        
    return 1