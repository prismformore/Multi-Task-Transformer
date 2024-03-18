
import os

PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]
pretrained_path = 'PRETRAINED_PATH'
db_root = 'DATASET_ROOT_DIR'

db_names = {'PASCALContext': 'PASCALContext'}
db_paths = {}
for database, db_p in db_names.items():
    db_paths[database] = os.path.join(db_root, db_p)
