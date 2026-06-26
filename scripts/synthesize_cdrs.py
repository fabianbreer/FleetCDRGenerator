import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(parent_dir, 'src'))
sys.path.append(os.getcwd()+'/src') 
from dataloader import *
from cluster_tree_model import *
from cdr_generator import *

# Define paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(parent_dir, "config.json")
 
def load_config():
    with open(config_path, "r") as file:
        return json.load(file)
    
def translate_cluster_name(cluster_name: str, dataloader, is_AC_model=None, is_DC_model=None) -> str:
    if is_AC_model and not is_DC_model:
       cluster_id = dataloader.cluster_id_from_label(cluster_name, 'AC')
    elif is_DC_model and not is_AC_model:
        cluster_id = dataloader.cluster_id_from_label(cluster_name, 'DC')
    return cluster_id

# Read config and translate cluster name
config = load_config()
experiment_name = config['experiment_name']
dataloader = Dataloader(base_path=parent_dir, experiment_name=experiment_name)
model = dataloader.load_model('cluster_models','cluster_tree_model.pkl')
cluster = translate_cluster_name(config['cluster'], dataloader, is_AC_model=model.AC_model, is_DC_model=model.DC_model)
method = config['method']
assert cluster in model.final_clusters, f"Cluster {config['cluster']} not found in final clusters {model.final_clusters}."

# Generate CDRs
cdr_generator = CDRGenerator(dataloader, cluster=cluster,  is_AC_model=model.AC_model, is_DC_model=model.DC_model)
cdrs = cdr_generator.generate_cdrs(method=method, n_cdrs_per_week=config['n_cdrs_per_week'], n_weeks=config['n_weeks'])

# save cdrs
output_dir = os.path.join(parent_dir, "results", experiment_name, "cdrs")
os.makedirs(output_dir, exist_ok=True)
cdrs.to_csv(os.path.join(output_dir, f"{config['cluster']}_cdrs_generated.csv"), index=False)

print(50*"-")
print(f"CDRs generated and saved for cluster {config['cluster']} in {output_dir}")
print(50*"-")
