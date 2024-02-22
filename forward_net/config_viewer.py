from trainer import Config
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(cur_dir,'..')
parent_parent_dir = os.path.join(cur_dir,'..', '..')
ancestor_dir = os.path.join(cur_dir,'..', '..', '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_parent_dir)
sys.path.insert(0, ancestor_dir)
import pickle
import pprint

path = os.path.join(cur_dir,'..','..','experiments')
name = 'single_transformer_8_layers_1_batch'
path = os.path.join(path, name, 'config.pkl')
print(path)
with open(path, 'rb') as f:
    config = pickle.load(f)
pprint.pprint(vars(config))