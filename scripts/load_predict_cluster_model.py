import sys
import os
sys.path.append(os.getcwd()+'/src') 
from dataloader import *
from cluster_tree_model import *

def load_config():
    """
    Load configuration from the config.json file.

    Returns
    -------
    dict
        Configuration dictionary loaded from the JSON file.
    """
    with open(config_path, "r") as file:
        return json.load(file)
    
config = load_config()
experment_name = config.get("experiment_name", "")
fleet_id = config.get("fleet_id", "")
cluster_map = config.get("cluster_map", {})
basepath = os.getcwd()
dataloader = Dataloader(basepath, experment_name)
model = dataloader.load_model('cluster_models','cluster_tree_model.pkl')
cluster = model.predict(fleet_id)
is_AC_or_DC = 'AC' if model.AC_model else 'DC' if model.DC_model else None
cluster = cluster_map.get(is_AC_or_DC).get(cluster, cluster)

print(50*"-")
print(f"Predicted cluster for fleet {fleet_id}: {cluster}")
print(50*"-")