import os

def get_img_with_ali(img_dir: str, return_index: bool=False, absolute_path: bool=False, return_img_path: bool=True):

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    res_json_list = []
    res_img_list = []
    res_index_list = []
    for img_ in img_files:
        corresponding_ali = img_.replace('.jpg', '_ali.json')
        index = int(img_.split('.')[0])
        ali_json_path = os.path.join(img_dir, corresponding_ali)
        if os.path.exists(ali_json_path):
            res_index_list.append(index)
            if absolute_path:
                res_json_list.append(ali_json_path)
                res_img_list.append(os.path.join(img_dir, img_))
            else:
                res_json_list.append(corresponding_ali)
                res_img_list.append(img_)

    if return_index:
        return sorted(res_index_list)
    else:
        if return_img_path:
            return sorted(res_img_list)
        else:
            return sorted(res_json_list)
        

def get_page_with_context(img_dir: str, return_index: bool=False, absolute_path: bool=False):
    img_with_ali = get_img_with_ali(img_dir, return_index=True)
    res_list = img_with_ali
    for i in img_with_ali[1:-2]:
        if i + 1 not in img_with_ali:
            res_list.append(i + 1)
        if i - 1 not in img_with_ali:
            res_list.append(i - 1)
    
    if img_with_ali[0] not in res_list:
        res_list.append(img_with_ali[0])
    if img_with_ali[-1] not in res_list:
        res_list.append(img_with_ali[-1])

    if return_index:
        return sorted(res_list)
    else:
        if absolute_path:
            return sorted([os.path.join(img_dir, f"{i:04d}.jpg") for i in sorted(res_list)])
        else:
            return sorted([f"{i:04d}.jpg" for i in sorted(res_list)])


if __name__ == "__main__":
    img_dir = "/11_data/zph/prog/qwenlayout/output/BiopsyInterpretationOftheCentralNervousSystem2ndEdition2018/images"
    print(len(get_page_with_context(img_dir, return_index=True)))
    print(len(get_img_with_ali(img_dir, return_index=True)))