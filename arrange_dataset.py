import os

DATASETS = [
    './dataset/train/',
    './dataset/test/',
]
CLASSES = [
    'normal_0',
    'top_left_1',
    'top_right_2',
    'bottom_left_3'
]
for dataset_path in DATASETS:
    for class_name in CLASSES:
        dir_path = dataset_path + class_name
        idx = 0
        for filename in os.listdir(dir_path):
            if 'png' in filename:
                os.rename(f'{dir_path}/{filename}', f'{dir_path}/dummy_{idx}.png')
                idx += 1
        idx = 0        
        for filename in os.listdir(dir_path):
            if 'png' in filename:
                os.rename(f'{dir_path}/{filename}', f'{dir_path}/{class_name}_{idx}.png')
                idx += 1
        print(f'[*] {dataset_path}{class_name}: {idx} files')
        with open(f'{dir_path}/index', 'w') as index_file:
            index_file.write(str(idx + 1))
