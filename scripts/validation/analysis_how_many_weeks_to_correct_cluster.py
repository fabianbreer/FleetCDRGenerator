import sys
import os
import pandas as pd
from tqdm import tqdm
sys.path.append(os.getcwd()+'/src') 
from dataloader import Dataloader
from cluster_tree_model import *  # required for unpickling ClusterTreeModel
from cdr_generator import CDRGenerator
from data_process import find_p_demand_for_each_charging_event_from_charging_data, generate_p_demand_occupancy_timeseries, process_weekly_data, process_weekday_weekend_data

method = 'gmm'
experiment_name = 'AC_model_gmm'
basepath = os.getcwd()
dataloader = Dataloader(basepath, experiment_name=experiment_name)
model = dataloader.load_model('cluster_models','cluster_tree_model.pkl')
model.dataloader = dataloader

n_weeks_list = []
correct_cluster_list = []
cluster_list = []
n_cdrs_per_week_list = []
wrong_cluster_list = []
repetition_list = []
if "AC" in experiment_name:
    model.AC_model = True
    model.DC_model = False
else:
    model.AC_model = False
    model.DC_model = True
for cluster in tqdm(model.final_clusters, desc="Processing clusters"):
    for n_cdrs_per_week in tqdm(range(5,101,5), desc="Processing sample sizes"):
        for repetition in range(1,6):
            repetition_list.append(repetition)
            correct_cluster_list.append(cluster)

            new_cluster = None
            n_weeks = 1
            n_cdrs_per_week_list.append(n_cdrs_per_week)
            wrong_cluster = -1

            while cluster != new_cluster:

                cdr_generator = CDRGenerator(dataloader, cluster, is_AC_model=model.AC_model, is_DC_model=model.DC_model)
                cdrs = cdr_generator.generate_cdrs(method, n_cdrs_per_week=n_cdrs_per_week, n_weeks=n_weeks)
                charging_data, charging_data_dict = find_p_demand_for_each_charging_event_from_charging_data(cdrs, basepath)

                p_demand_timeseries, charging_data = generate_p_demand_occupancy_timeseries(charging_data, charging_data_dict, 15, localize_dt=False)
                _ = process_weekly_data(p_demand_timeseries, dataloader.data_path, 'predict')
                _ = process_weekday_weekend_data(p_demand_timeseries, dataloader.data_path, 'predict')
                new_cluster = model.predict('predict')
                n_weeks += 1
                if n_weeks > 52:
                    wrong_cluster = new_cluster
                    break

            cluster_list.append(new_cluster)
            n_weeks_list.append(n_weeks)
            wrong_cluster_list.append(wrong_cluster)

            df = pd.DataFrame({'n_weeks_to_correct_cluster': n_weeks_list, 'correct_cluster': correct_cluster_list,'cluster_list':cluster_list,'n_cdrs_per_week': n_cdrs_per_week_list, 'wrong_cluster': wrong_cluster_list, 'repetition':repetition_list})
            df.to_csv(os.path.join(basepath, 'results', f'{experiment_name}','analysis', f'n_weeks_to_correct_cluster_new_{method}.csv'), index=False)

