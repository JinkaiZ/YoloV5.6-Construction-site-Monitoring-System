from pathlib import Path

import yaml

cfg = '..\hardhatDetection\yolov5\dataset.yaml'
yaml_file = Path(cfg).name


# users = {'path': '../datasets/data', 'train': 'images', 'val': 'images', 'nc': 3, 'names': [ 'person',  'non hardhat wearing', 'wearing hardhat']}
# with open(cfg,'w') as f:
#         yaml.dump(users, f)


with open(cfg,) as f:
        yaml = yaml.safe_load(f)
names = yaml['names']
print(names)