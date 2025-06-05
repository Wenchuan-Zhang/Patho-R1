import os
import json
import cv2

def cropping(json_source_dir: str, img_source_dir: str, img_target_dir: str):

    os.makedirs(img_target_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_source_dir) if f.endswith('_ali.json')]
    # img_index = [index.split('_')[0] for index in json_files]

    for json_file in json_files:
        json_path = os.path.join(json_source_dir, json_file)
        img_file = json_file.replace('_ali.json', '.jpg')
        img_path = os.path.join(img_source_dir, img_file)

        img = cv2.imread(img_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)

        for i, fig in enumerate(analysis):

            x1, y1, x2, y2 = [int(coord * 2) for coord in fig['bbox']]
            cropped_img = img[y1:y2, x1:x2]
            crop_filename = f"{img_file[:-4]}_fig{i}.jpg"
            crop_path = os.path.join(img_target_dir, crop_filename)
            cv2.imwrite(crop_path, cropped_img)
            fig['figure_path'] = crop_path

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=4, ensure_ascii=False)


def cropping_main_text(img_dir, json_dir):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    # json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    json_paths = []
    for img_file in img_files:
        json_file = img_file.replace('.jpg', '.json')
        json_path = os.path.join(json_dir, json_file)
        cropped_text_path = json_path.replace('.json', '')
        if os.path.exists(cropped_text_path):
            pass
        else:
            os.makedirs(cropped_text_path, exist_ok=True)
            json_paths.append(json_path)

    to_be_ocr = []
    for json_path in json_paths:
        img_file = json_path.split('/')[-1].replace('.json', '.jpg')
        img_path = os.path.join(img_dir, img_file)
        crop_folder = json_path.replace('.json', '')
        os.makedirs(crop_folder, exist_ok=True)
        img = cv2.imread(img_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)

        analysis = json.loads(analysis)
        for i, box in enumerate(analysis):
            
            if box['class'] == 1 or box['class'] == '1':
                bbox = box.get('box', {})
                x1, y1, x2, y2 = bbox.get('x1'), bbox.get('y1'), bbox.get('x2'), bbox.get('y2')
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                crop_filename = f"{img_file[:-4]}_fig{i}.jpg"
                crop_path = os.path.join(crop_folder, crop_filename)
                cv2.imwrite(crop_path, cropped_img)
                to_be_ocr.append(crop_path)
                
    return to_be_ocr


# if __name__ == "__main__":
#     cropping_main_text("output/images"