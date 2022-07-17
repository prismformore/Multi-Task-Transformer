import os

db_root = 'your_dataset_directory'
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[0]

db_names = {'PASCALContext', 'NYUDv2'}
db_paths = {}
for database in db_names:
    db_paths[database] = os.path.join(db_root, database)