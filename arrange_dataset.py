import os

DATASETS = [
    './dataset/train/',
]
CLASSES = [
    'normal_0',
    'top_left_1'
]
for dataset_path in DATASETS:
    for class_name in CLASSES:
        dir_path = dataset_path + class_name
        for idx, filename in enumerate(os.listdir(dir_path)):
            if 'png' in filename:
                os.rename(f'{dir_path}/{filename}', f'{dir_path}/dummy_{idx}.png')
        for idx, filename in enumerate(os.listdir(dir_path)):
            if 'png' in filename:
                os.rename(f'{dir_path}/{filename}', f'{dir_path}/{class_name}_{idx}.png')
