import sys
import os
sys.path.append(os.getcwd()+'/src') 
from dataloader import *
from cluster_tree_model import *

#############How To Use#################################################################################################################
# The Charge Detail Records (CDRs) that will be analyzed by this script need to be in the folder: data/power_demands/cdrs/
# The CDRs should be in CSV format and named as <fleet_id>_cdrs.csv, where <fleet_id> is the identifier of the fleet.
# The script will load the model from the folder: data/cluster_models/cluster_tree_model.pkl
# The script will output the predicted cluster for the given fleet_id based on the model and the configuration provided in config.json
########################################################################################################################################

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
experiment_name = config.get("experiment_name", "")
fleet_id = config.get("fleet_id", "")
cluster_map = config.get("cluster_map", {})
basepath = os.getcwd()
dataloader = Dataloader(basepath, experiment_name)
model = dataloader.load_model('cluster_models','cluster_tree_model.pkl')
cluster = model.predict(fleet_id)
is_AC_or_DC = 'AC' if model.AC_model else 'DC' if model.DC_model else None
cluster = cluster_map.get(is_AC_or_DC).get(cluster, cluster)

print(50*"-")
print(f"Predicted cluster for fleet {fleet_id}: {cluster}")
print(50*"-")