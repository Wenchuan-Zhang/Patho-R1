import os
import json
import re
import prompts
from tqdm import tqdm
import asyncio
from layout_async import send_api_request_async
from typing import List, Dict, Tuple, Optional, Any


def update_prompt():
    global system_prompt, user_prompt
    if prompts.language == "zh":
        system_prompt = prompts.system_prompt_inline_zh
        user_prompt = prompts.prompt_inline_zh
    else:
        system_prompt = prompts.system_prompt_inline_en
        user_prompt = prompts.prompt_inline_en

async def process_single_image(img_file, img_dir, api_url):
    
    # print(f"started processing {img_file}")
    img_path = os.path.join(img_dir, img_file)
    index = img_file.split('.')[0]
    json_path = os.path.join(img_dir, f"{index}_ali.json")
    # if not os.path.exists(json_path):
    #     return

    # with open(json_path, 'r',) as f:
    #     current_json = json.load(f)

    # if isinstance(current_json, list) and current_json and isinstance(current_json[0], dict):
    #     if "inline_ref" in current_json[0]:
    #         return
    
    prev_content = get_page_content(img_dir, index, -1)
    current_content = get_page_content(img_dir, index, 0)
    next_content = get_page_content(img_dir, index, 1)
    reference = f"{prev_content}\n{current_content}\n{next_content}"

    with open(json_path, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    json_str = json.dumps(analysis, indent=2, ensure_ascii=False)

    prompt=user_prompt.format(json_text=json_str, main_text=reference)

    payload = {
        "model": "qwen-max",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]}
        ],
        "max_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.8,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }

    await asyncio.sleep(1)

    try:
        content = await send_api_request_async(payload, api_url)
    except Exception as e:
        print(e)
        with open(os.path.join(os.path.dirname(img_path), 'error.txt'), 'a', encoding='utf-8') as f:
            f.write(f"{img_path}\n")
        return
    try:
        if content:
            # print(f"recieved content for {img_file}")
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content
            
            analysis_model = json.loads(json_str)

            for dict_ori, dict_added in zip(analysis, analysis_model):
                dict_ori['inline_ref'] = dict_added['inline_ref']

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=4, ensure_ascii=False)

    except Exception as e:
        print(e)
        with open(os.path.join(os.path.dirname(img_path), 'error.txt'), 'a', encoding='utf-8') as f:
            f.write(f"{img_path}\n")

    # print(f"finished processing {img_file}")


# async def inline_ref_async(img_dir: str, api_url, concurrency=5) -> None:
#     img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
#     semaphore = asyncio.Semaphore(concurrency)
#     tasks = [process_single_image(img_file, img_dir, api_url, semaphore) for img_file in img_files]
#     await asyncio.gather(*tasks)

async def inline_ref_async(img_dir: str, api_url, concurrency=10) -> None:

    update_prompt()
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    filtered_img_files = [f for f in img_files
                                 if os.path.exists(os.path.join(img_dir, f.replace('.jpg', '_ali.json')))]
    img_files = []
    for img_ in filtered_img_files:
        with open(os.path.join(img_dir, img_.replace('.jpg', '_ali.json')), 'r', encoding='utf-8') as j:
            current_json = json.load(j)

        if isinstance(current_json, list) and current_json and isinstance(current_json[0], dict):
            if "inline_ref" not in current_json[0]:
                img_files.append(img_)

    img_files.sort()

    semaphore = asyncio.Semaphore(concurrency)

    async def wrapped_task(img_file):
        async with semaphore:
            # print(f"trying to start processing {img_file}")
            await process_single_image(img_file, img_dir, api_url)
            progress_bar.update(1)
            # print(f"finished processing {img_file}")

    progress_bar = tqdm(total=len(img_files), desc="Processing images", unit="img")
    tasks = [wrapped_task(img_file) for img_file in img_files]
    await asyncio.gather(*tasks)
    progress_bar.close()


def get_page_content(img_dir: str, index: str, offset: int) -> str:
    """获取指定偏移量的页面内容"""
    try:
        page_num = int(index)
        target_file = f"{page_num + offset:04d}_main.txt"
        target_path = os.path.join(img_dir, target_file)
        if os.path.exists(target_path):
            with open(target_path, 'r', encoding='utf-8') as f:
                return f.read()
    except ValueError:
        pass
    return ""