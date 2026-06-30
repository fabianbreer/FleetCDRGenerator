import sys
import os
from tqdm import tqdm
import json
sys.path.append(os.getcwd()+'/src') 
from dataloader import Dataloader
from cluster_tree_model import * # required for unpickling ClusterTreeModel
from cdr_generator import CDRGenerator

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(parent_dir, "config.json")
 
def load_config():
    with open(config_path, "r") as file:
        return json.load(file)
    
config = load_config()
experiment_name = config['experiment_name']
basepath = os.getcwd()
dataloader = Dataloader(basepath, experiment_name=experiment_name)
model = dataloader.load_model('cluster_models','cluster_tree_model.pkl')
model.dataloader = dataloader
final_clusters = model.final_clusters

print("Final clusters to process:", final_clusters)
if "AC" in experiment_name:
    model.AC_model = True
    model.DC_model = False
else:
    model.AC_model = False
    model.DC_model = True
for cluster in tqdm(final_clusters, desc="Processing clusters"):
    cdr_generator = CDRGenerator(dataloader, cluster, is_AC_model=model.AC_model, is_DC_model=model.DC_model)
    _ = cdr_generator.fit_base_model(max_components=10)

