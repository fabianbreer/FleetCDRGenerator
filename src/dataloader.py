import pickle
import os
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import json
from collections import Counter

class Dataloader():
    def __init__(self, base_path, experiment_name=None):
        """
        Initialize the Dataloader.

        This class centralizes paths and file I/O for models, results and
        preprocessed data used across the project. It ensures expected
        directories exist for the given experiment and exposes helpers to
        save/load models and CSV artifacts.

        Parameters
        ----------
        base_path : str
            Repository root directory (where `data/`, `models/`, `results/` live).
        experiment_name : str or None, optional
            Name of the experiment used to separate model and result folders.
        """
        self.base_path = base_path
        self.data_path = os.path.join(base_path, 'data/')
        self.model_path = os.path.join(base_path, 'models/')
        self.experiment_name = experiment_name if experiment_name else 'default_experiment'

        if not os.path.exists(os.path.join(self.model_path, self.experiment_name)):
            os.makedirs(os.path.join(self.model_path, self.experiment_name))
            os.makedirs(os.path.join(self.model_path, self.experiment_name, 'generator_models'))
            os.makedirs(os.path.join(self.model_path, self.experiment_name, 'cluster_models'))
            os.makedirs(os.path.join(self.model_path, self.experiment_name, 'cluster_models', 'submodels'))

        if not os.path.exists(os.path.join(self.base_path, 'results', self.experiment_name)):
            os.makedirs(os.path.join(self.base_path,'results', self.experiment_name, 'cdrs'))

    def save_model(self, model, subfolder, filename):
        """
        Save a Python object (model) to the experiment model folder using pickle.

        Parameters
        ----------
        model : object
            Python object to persist (typically a trained estimator).
        subfolder : str
            Subfolder under the experiment model directory (e.g. 'cluster_models').
        filename : str
            Filename to write (including extension, e.g. 'model.pkl').
        """
        with open(f'{self.model_path}/{self.experiment_name}/{subfolder}/{filename}', 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, subfolder, filename):
        """
        Load a pickled model object from the experiment model folder.

        Parameters
        ----------
        subfolder : str
            Subfolder under the experiment model directory where the file resides.
        filename : str
            Filename to load (including extension).

        Returns
        -------
        object
            Unpickled Python object (typically a trained estimator).
        """
        with open(f'{self.model_path}/{self.experiment_name}/{subfolder}/{filename}', 'rb') as f:
            model = pickle.load(f)
        model.dataloader = self
        return model
    
    def save_base_model(self, model, filename):
        """
        Save a base model object to the experiment model folder under 'base_models/'.

        Parameters
        ----------
        model : object
            Python object to persist (e.g. a trained base model).
        filename : str
            Filename to write (including extension, e.g. 'base_model.pkl').
        """
        path = f'{self.model_path}/{self.experiment_name}/base_models/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, filename), 'wb') as f:
            pickle.dump(model, f)

    def save_submodel(self, model, subfolder,filename):
        """
        Save a model into the `submodels` directory under an experiment subfolder.

        Parameters
        ----------
        model : object
            Model object to persist.
        subfolder : str
            Subfolder under the experiment models folder.
        filename : str
            Filename for the saved submodel.
        """
        with open(f'{self.model_path}/{self.experiment_name}/{subfolder}/submodels/{filename}', 'wb') as f:
            pickle.dump(model, f)

    def load_ts_submodel(self, subfolder, cluster_name):
        """
        Load a time-series submodel for a cluster. The filename is constructed
        as `{cluster_name}_model.pkl` and expected under `submodels/`.

        Parameters
        ----------
        subfolder : str
            Subfolder under the experiment models where submodels are stored.
        cluster_name : str
            Cluster identifier used to compose the submodel filename.

        Returns
        -------
        object
            The unpickled model object.
        """
        filename = f'{cluster_name}_model.pkl'
        with open(f'{self.model_path}/{self.experiment_name}/{subfolder}/submodels/{filename}', 'rb') as f:
            model = pickle.load(f)
        # model = TimeSeriesKMeans.from_pickle(f'{self.model_path}/{subfolder}/submodels/{filename}')
        return model

    def load_submodel(self, subfolder, cluster_name):
        """
        Load a submodel saved under `submodels/` for a given cluster name.

        Parameters
        ----------
        subfolder : str
            Model subfolder name.
        cluster_name : str
            Cluster id used to build the filename `{cluster_name}_model.pkl`.

        Returns
        -------
        object
            Unpickled model object.
        """
        filename = f'{cluster_name}_model.pkl'
        with open(f'{self.model_path}/{self.experiment_name}/{subfolder}/submodels/{filename}', 'rb') as f:
            model = pickle.load(f)
        return model
    
    def get_all_models(self, subfolder):
        """
        List available submodel prefixes present in a subfolder's `submodels` directory.

        Parameters
        ----------
        subfolder : str
            Subfolder name under the experiment models directory.

        Returns
        -------
        list of str
            List of model name prefixes (before the first underscore) found in the directory.
        """
        return [f.split('_')[0] for f in os.listdir(f'{self.model_path}/{self.experiment_name}/{subfolder}/submodels') if os.path.isfile(os.path.join(f'{self.model_path}/{self.experiment_name}/{subfolder}/submodels', f))]

    def get_in_array(self, timeframe, AC_model=True, DC_model=False):
        """
        Load precomputed time series arrays for clustering/analysis.

        This helper loads CSVs from `data/power_demands_analysis/{timeframe}`
        (filtered by AC/DC if requested), normalizes each series by its max,
        pads series to a common length and returns a NumPy array and the list
        of csv filenames used.

        Parameters
        ----------
        timeframe : str
            Timeframe identifier (e.g. 'weekly_average', 'weekday_average').
        AC_model : bool, optional
            When True, prefer AC-mode files.
        DC_model : bool, optional
            When True, prefer DC-mode files.

        Returns
        -------
        tuple
            (input_array, csv_file_names) where `input_array` is an ndarray of
            normalized/padded time series and `csv_file_names` is the list of files used.
        """
        if DC_model:
            assert not AC_model, 'You can not use both AC and DC model at the same time.'

        if AC_model:
            assert not DC_model, 'You can not use both AC and DC model at the same time.'

        if timeframe == 'weekday_average':
            timeframe_new = 'weekday_weekend_average'
        else:
            timeframe_new = timeframe

        input_array = []

        # csv_file_names = os.listdir(f'{self.data_path}power_demands_analysis/{timeframe_new}')
        csv_file_names = filter_out_csv_file_names(self.data_path, AC_model=AC_model, DC_model=DC_model, timeframe=timeframe_new)
        for i, file in enumerate(csv_file_names):
            path_file = f'{self.data_path}power_demands_analysis/{timeframe_new}/{file}'  # path to CSV
            df = pd.read_csv(path_file)
            if df.empty:
                continue
            # ensure no negative entries
            assert df['Average_power_demand'].any()>=0, 'Fleet Clustering: Average Power Array cant have negative numbers.'

            input_array.append(df['Average_power_demand'].values/df['Average_power_demand'].max())

        input_array = np.array(input_array)
        max_length = max(len(x) for x in input_array)
        input_array = np.array([np.pad(x, (0, max_length - len(x)), 'constant') for x in input_array])
        input_array = np.nan_to_num(input_array, nan=0.0)
        if timeframe == 'weekday_average':
            input_array = input_array[:,0:24]

        return input_array, csv_file_names
    
    # ----------------------
    # Config / cluster map helpers
    # ----------------------
    def _load_config(self):
        """Load and cache the project's top-level config.json.

        Returns
        -------
        dict
            Parsed JSON from <base_path>/config.json
        """
        if hasattr(self, '_config_cache') and self._config_cache is not None:
            return self._config_cache
        config_path = os.path.join(self.base_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found at {config_path}")
        with open(config_path, 'r') as f:
            import json
            self._config_cache = json.load(f)
        return self._config_cache

    def get_cluster_map(self):
        """Return the cluster_map dict from config.json.

        Example return value:
        { 'AC': {'00010': 'A1', ...}, 'DC': {...} }
        """
        cfg = self._load_config()
        return cfg.get('cluster_map', {})

    def reverse_cluster_map(self):
        """Return a reversed cluster map: label -> cluster id for each model (AC/DC).

        Example:
        { 'AC': {'A1': '00010', ...}, 'DC': {'C1': '0121', ...} }
        """
        cm = self.get_cluster_map()
        reversed_map = {}
        for model, mapping in cm.items():
            # invert mapping; if duplicates occur the last wins
            reversed_map[model] = {v: k for k, v in mapping.items()}
        return reversed_map

    def cluster_label(self, cluster_id, model='AC'):
        """Return the human-readable label for a cluster id (e.g. 'A1').

        Parameters
        ----------
        cluster_id : str
            Internal cluster id (key in cluster_map[model]).
        model : str, optional
            Either 'AC' or 'DC'. Default 'AC'.
        """
        cm = self.get_cluster_map()
        return cm.get(model, {}).get(cluster_id)

    def cluster_id_from_label(self, label, model='AC'):
        """Return the internal cluster id for a given human-readable label.

        Example: cluster_id_from_label('A1', model='AC') -> '00010'
        """
        rev = self.reverse_cluster_map()
        return rev.get(model, {}).get(label)
    
    def save_csv_files(self, clusters, names, ids, variances, distances):
        """
        Save cluster assignment CSVs and update cluster sizes metadata.

        For each unique cluster label in `clusters`, a CSV mapping `Fleet_Id`
        to cluster id will be written. Additionally, the method appends cluster
        size/variance/distance rows to `cluster_sizes.csv` under results.

        Parameters
        ----------
        clusters : array-like
            Array of cluster labels assigned to `names`.
        names : sequence of str
            Filenames corresponding to each sample (used to extract fleet ids).
        ids : sequence of str
            Cluster id strings to use when saving cluster CSVs.
        variances : sequence
            Per-cluster variance values (nested list/array expected as in upstream code).
        distances : sequence
            Per-cluster DTW distance values (nested list/array expected).
        """

        folder = self.base_path+f'/results/{self.experiment_name}/csv_files/'
        os.makedirs(folder, exist_ok=True)

        for i, c in enumerate(np.unique(clusters)):
            fleet_ids = [name.split('_')[0] for name in names[clusters == c]]
            df_cluster_names = pd.DataFrame({'Cluster': ids[i], 'Fleet_Id': fleet_ids})

            df_cluster_names.to_csv(folder+'cluster_names_{}.csv'.format(ids[i]), index=False)

        cluster_sizes_path = folder + 'cluster_sizes.csv'
        if not os.path.exists(cluster_sizes_path):
            df_cluster_sizes = pd.DataFrame(columns=['Cluster', 'Size', 'Variance', 'Distance', 'Parent'])
            df_cluster_sizes.to_csv(cluster_sizes_path, index=False)
        df_cluster_sizes = pd.read_csv(folder+'cluster_sizes.csv', dtype={'Cluster': str})
        df_cluster_sizes = pd.concat([df_cluster_sizes, pd.DataFrame({'Cluster': ids, 'Size': [len(clusters[clusters == c]) for c in np.unique(clusters)], 'Variance': [variances[0][i] for i in np.unique(clusters)], 'Distance': [distances[0][i] for i in np.unique(clusters)]})], axis=0)
        df_cluster_sizes.to_csv(folder+'cluster_sizes.csv'.format(folder), index=False)

    
    def save_gmm_model(self, model, cluster_name, day):
        """
        Persist a GMM model used for start-time generation for a cluster/day.

        Parameters
        ----------
        model : object
            The GMM object to serialize.
        cluster_name : str
            Cluster identifier used in the filename.
        day : int
            Day index (0-6) used in the filename.
        """
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/{cluster_name}_gmm_start_time_model_day_{day}.pkl', 'wb') as f:
            pickle.dump(model, f)

    def save_gmm_model_energy(self, model, cluster_name, week_hour):
        """
        Save an energy GMM model for a cluster at a given week_hour bin.

        Parameters
        ----------
        model : object
            GMM energy model to save.
        cluster_name : str
            Cluster identifier.
        week_hour : int
            Week-hour bin used in the filename.
        """
        path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/energy/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{cluster_name}_gmm_energy_model_weekhour_{week_hour}.pkl'), 'wb') as f:
            pickle.dump(model, f)


    def save_energy_scaler(self, scaler, cluster_name, week_hour):
        """
        Persist an energy scaler object for a cluster/week_hour.

        Parameters
        ----------
        scaler : object
            Scaler (e.g. sklearn scaler) to save.
        cluster_name : str
            Cluster id for filename.
        week_hour : int
            Week-hour bin index used in filename.
        """
        path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/scalers/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{cluster_name}_energy_scaler_weekhour_{week_hour}.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    def save_duration_scaler(self, scaler, cluster_name, week_hour):
        """
        Persist a duration scaler object for a cluster/week_hour.

        Parameters
        ----------
        scaler : object
            Scaler instance to save.
        cluster_name : str
            Cluster id used in filename.
        week_hour : int
            Week-hour bin index used in filename.
        """
        path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/scalers/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{cluster_name}_duration_scaler_weekhour_{week_hour}.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    def save_duration_scaler_base(self, scaler, cluster_name, day):
        """
        Persist a duration scaler object for a cluster/day.

        Parameters
        ----------
        scaler : object
            Scaler instance to save.
        cluster_name : str
            Cluster id used in filename.
        day : int
            Day index (0-6) used in filename.
        """
        path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/scalers/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{cluster_name}_duration_scaler_day_{day}.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    
    def load_duration_scaler_base(self, cluster_name, day):
        """
        Load a previously saved duration scaler for a cluster/day.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier.
        day : int
            Day index (0-6) whose scaler is required.

        Returns
        -------
        object
            The loaded scaler object.
        """
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/scalers/{cluster_name}_duration_scaler_day_{day}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    
    def save_energy_scaler_base(self, scaler, cluster_name, day):
        """
        Persist an energy scaler object for a cluster/day.

        Parameters
        ----------
        scaler : object
            Scaler instance to save.
        cluster_name : str
            Cluster id used in filename.
        day : int
            Day index (0-6) used in filename.
        """
        path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/scalers/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{cluster_name}_energy_scaler_day_{day}.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    def get_fleet_meta_data(self, final_clusters=None):
        """
        Load the fleet meta data from the CSV file.
        :return: DataFrame containing the fleet meta data.
        """
        fleet_meta_data_path = f'{self.data_path}/power_demands_analysis/daily_average/'
        files = [f for f in os.listdir(fleet_meta_data_path) if f.startswith('1')]
        files = [f.split('_')[0] for f in files]

        clusters = []
        if final_clusters is not None:
            for file in files:
                is_none = True
                for cluster in final_clusters:
                    cluster_names = pd.read_csv(f'{self.base_path}/results/csv_files/cluster_names_{cluster}.csv', dtype={'Cluster': str, 'Fleet_Id': str})
                    if file in cluster_names['Fleet_Id'].values:
                        clusters.append(cluster)
                        is_none = False
                        break
                if is_none:
                    clusters.append(None)

            df = pd.DataFrame({'Fleet_Id': files, 'Cluster': clusters})
        else:
            df = pd.DataFrame({'Fleet_Id': files, 'Cluster': [None] * len(files)})
            
        
        return df
    
    def get_cdrs_for_fleet(self, fleet_id):
        """
        Load the CDRs for a given fleet ID.
        :param fleet_id: Fleet ID.
        :return: DataFrame containing the CDRs for the specified fleet ID.
        """
        path = f'{self.data_path}/power_demands/cdrs/{fleet_id}_cdrs.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.start_time = pd.to_datetime(df.start_time)
            df.end_time = pd.to_datetime(df.end_time)
            df.sort_values(by='start_time', inplace=True)
            return df
        else:
            return pd.DataFrame()

    def get_n_cdrs_per_week_for_fleet(self, fleet_id, ac_only=False, dc_only=False):
        """
        Load the number of CDRs per week for a given fleet ID.
        :param fleet_id: Fleet ID.
        :return: Number of CDRs per week for the specified fleet ID.
        """
        assert ac_only*dc_only == 0, 'You can not use both AC and DC only at the same time.'
        path = f'{self.data_path}/power_demands/cdrs/{fleet_id}_cdrs.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            if ac_only:
                df = df[df.max_socket_power <= 22]
            elif dc_only:
                df = df[df.max_socket_power > 22]
            df.start_time = pd.to_datetime(df.start_time)
            df.sort_values(by='start_time', inplace=True)
            earliest_date = df.start_time.min()
            latest_date = df.start_time.max()
            n_weeks = (latest_date - earliest_date).days // 7 + 1

            n_cdrs_per_week = df.shape[0] // n_weeks
            return n_cdrs_per_week



    def load_energy_scaler_base(self, cluster_name, day):
        """
        Load a previously saved energy scaler for a cluster/day.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier.
        day : int
            Day index (0-6) whose scaler is required.

        Returns
        -------
        object
            The loaded scaler object.
        """
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/scalers/{cluster_name}_energy_scaler_day_{day}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    def save_duration_scaler(self, scaler, cluster_name, week_hour):
        """
        Persist a duration scaler object for a cluster/week_hour.

        Parameters
        ----------
        scaler : object
            Scaler instance to save.
        cluster_name : str
            Cluster id used in filename.
        week_hour : int
            Week-hour bin index used in filename.
        """
        path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/scalers/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{cluster_name}_duration_scaler_weekhour_{week_hour}.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    
    def load_energy_scaler(self, cluster_name, week_hour):
        """
        Load a previously saved energy scaler for a cluster/week_hour.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier.
        week_hour : int
            Week-hour bin whose scaler is required.

        Returns
        -------
        object
            The loaded scaler object.
        """
        week_hour = (int(week_hour/4))%167
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/scalers/{cluster_name}_energy_scaler_weekhour_{week_hour}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    
    def load_duration_scaler(self, cluster_name, week_hour):
        """
        Load a previously saved duration scaler for a cluster/week_hour.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier.
        week_hour : int
            Week-hour bin index.

        Returns
        -------
        object
            The loaded scaler object.
        """
        week_hour = (int(week_hour/4))%167
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/scalers/{cluster_name}_duration_scaler_weekhour_{week_hour}.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    def load_gmm_model_energy(self, cluster_name, week_hour):
        """
        Load a saved energy GMM model for a cluster at a particular week_hour.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier.
        week_hour : int
            Week-hour index used when saving the model.

        Returns
        -------
        object
            The unpickled GMM model.
        """
        week_hour = (int(week_hour/4))%167
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/energy/{cluster_name}_gmm_energy_model_weekhour_{week_hour}.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    def save_gmm_model_duration(self, model, cluster_name, week_hour, energy_bin):
        """
        Save a duration GMM model for a cluster/week_hour/energy_bin.

        Parameters
        ----------
        model : object
            GMM duration model to persist.
        cluster_name : str
            Cluster identifier.
        week_hour : int
            Week-hour index used in naming.
        energy_bin : int
            Energy bin identifier used in naming.
        """
        path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/duration/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{cluster_name}_gmm_duration_model_weekhour_{week_hour}_energybin_{energy_bin}.pkl'), 'wb') as f:
            pickle.dump(model, f)

    def load_gmm_model_duration(self, cluster_name, week_hour, quantity_in_wh):
        """
        Load a duration GMM model for a cluster, selecting an energy bin based on quantity.

        Parameters
        ----------
        cluster_name : str
            Cluster id.
        week_hour : int
            Week-hour index.
        quantity_in_wh : numeric
            Quantity in Wh used to compute an energy bin selection.

        Returns
        -------
        object
            Loaded GMM duration model.
        """
        week_hour = (int(week_hour/4))%167
        quantity_in_wh = max(10000, round(quantity_in_wh, -4))
        energy_bin = int(min([quantity_in_wh,140000]))
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/duration/{cluster_name}_gmm_duration_model_weekhour_{week_hour}_energybin_{energy_bin}.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    
    def load_gmm_model_duration_parallel(self, cluster_name, week_hour):
        """
        Load a duration GMM model (parallel version) that was saved per week_hour.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier.
        week_hour : int
            Week-hour index.

        Returns
        -------
        object
            Loaded model object.
        """
        week_hour = (int(week_hour/4))%167
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/duration/{cluster_name}_gmm_duration_model_weekhour_{week_hour}.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    def save_gmm_model_duration_parallel(self, model, cluster_name, week_hour):
        """
        Save the parallel duration GMM model for a cluster/week_hour.

        Parameters
        ----------
        model : object
            Model to persist.
        cluster_name : str
            Cluster identifier.
        week_hour : int
            Week-hour index.
        """
        path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/duration/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{cluster_name}_gmm_duration_model_weekhour_{week_hour}.pkl'), 'wb') as f:
            pickle.dump(model, f)

    def load_gmm_model(self, cluster_name, day):
        """
        Load a start-time GMM model for a cluster and day index.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier.
        day : int
            Day of week index (0-6).

        Returns
        -------
        object
            Loaded GMM start-time model.
        """
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/{cluster_name}_gmm_start_time_model_day_{day}.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

    def save_day_probabilities(self, day_probabilities, cluster_name):
        """
        Save a pandas Series or mapping of day probabilities for a cluster to JSON.

        Parameters
        ----------
        day_probabilities : pandas.Series or dict
            Day -> probability mapping to save.
        cluster_name : str
            Cluster identifier.
        """
        folder = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models'

        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/{cluster_name}_day_probabilities.json'
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump({}, f)
        with open(file_path, 'w') as f:
            json.dump(day_probabilities.to_dict(), f)

    def load_day_probabilities(self, cluster_name):
        """
        Load day probabilities JSON for a cluster and return as a DataFrame.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ['day', 'probability'] sorted by index.
        """
        with open(f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/{cluster_name}_day_probabilities.json', 'r') as f:
            day_probabilities = json.load(f)
        return pd.DataFrame.from_dict(day_probabilities, orient='index', columns=['probability']).reset_index().rename(columns={'index': 'day'})

        
    def load_mean_power_profile(self, cluster):
        """
        Retrieve the precomputed mean power profile for a cluster.

        Parameters
        ----------
        cluster : str
            Cluster identifier string (e.g. '01'). The function expects
            profile files named like `{prefix}_profile_Cluster_{suffix}.csv`.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            If available, returns the 'mean' column as a Series; otherwise an empty DataFrame.
        """
        mean_power_profile_path = f'{self.base_path}/results/csv_files/{cluster[0:-1]}_profile_Cluster_{cluster[-1]}.csv'
        if os.path.exists(mean_power_profile_path):
            df = pd.read_csv(mean_power_profile_path)['mean']
            return df
        else:
            return pd.DataFrame()
        
        
    def load_example_cdrs(self, n_samples=1000):
        """
        Generate a small synthetic DataFrame of example CDRs for testing/inspection.

        Parameters
        ----------
        n_samples : int, optional
            Number of synthetic CDR rows to generate (default 1000).

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns: 'start_time', 'duration_sec', 'quantity_in_wh', 'cluster'.
        """
        return pd.DataFrame({
            "start_time": np.random.normal(12, 3, n_samples),
            "duration_sec": np.random.normal(2, 0.5, n_samples),
            "quantity_in_wh": np.random.normal(30, 5, n_samples),
            "cluster": np.random.choice([1,2,3,4,5,6,7,8,9,10], n_samples)
    })

    def save_gmm_base_model(self, model, cluster_name, model_type, day=None):
        """
        Save a base GMM model for a cluster and model type (start_time, energy, duration).

        Parameters
        ----------
        model : object
            The GMM model to save.
        cluster_name : str
            Cluster identifier used in the filename.
        model_type : str
            Type of the model ('start_time', 'energy', 'duration') used in naming.
        day : int, optional
            Day of the week (0-6) for which to save the model. If None, save as overall model.
        """
        path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/'
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'{cluster_name}_gmm_base_model_{model_type}_day_{day}.pkl'), 'wb') as f:
            pickle.dump(model, f)

    def load_original_cdrs(self, cluster, is_AC_model=True, is_DC_model=False):
        """
        Load the original CDRs for a given cluster.
        :param cluster: Cluster name or ID.
        :return: DataFrame containing the original CDRs for the specified cluster.
        """
        cluster_names = pd.read_csv(f'{self.base_path}/results/{self.experiment_name}/csv_files/cluster_names_{cluster}.csv')
        all_cdrs = []
        for name in cluster_names['Fleet_Id']:
            file_path = f'{self.data_path}power_demands/cdrs/{name}_cdrs.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
                if is_AC_model:
                    df = df[df.max_socket_power <= 22]
                if is_DC_model:
                    df = df[df.max_socket_power > 22]
                all_cdrs.append(df)
        if all_cdrs:
            all_dfs = pd.concat(all_cdrs, ignore_index=True)
            all_dfs['cluster']=cluster
            return all_dfs
        else:
            return pd.DataFrame()
        
    def save_gmm_metrics(self, metrics:dict, cluster_name, day=None):
        """
        Save the GMM metrics for a given cluster.
        :param df: DataFrame containing the GMM metrics.
        :param cluster_name: Name of the cluster.
        :param day: Day of the week (0-6). If None, save as overall metrics.
        """
        folder = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/{cluster_name}_gmm_metrics{"_day_"+str(day) if day is not None else ""}.json'
        metrics['grid_search'] = metrics['grid_search'].to_dict(orient="records")
        with open(file_path, 'w') as f:
            json.dump(metrics, f)

    def save_gmm_metrics_energy(self, metrics, cluster_name, week_hour):
        """
        Save the GMM metrics for energy for a given cluster.
        :param df: DataFrame containing the GMM metrics.
        :param cluster_name: Name of the cluster.
        :param week_hour: Week hour.
        """
        folder = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/energy'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/{cluster_name}_gmm_metrics_energy_weekhour_{week_hour}.json'
        metrics['grid_search'] = metrics['grid_search'].to_dict(orient="records")
        with open(file_path, 'w') as f:
            json.dump(metrics, f)

    def save_gmm_metrics_duration_parallel(self, metrics, cluster_name, week_hour):
        """
        Save the GMM metrics for duration for a given cluster.
        :param df: DataFrame containing the GMM metrics.
        :param cluster_name: Name of the cluster.
        :param week_hour: Week hour.
        """
        folder = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/duration'
        os.makedirs(folder, exist_ok=True)
        file_path = f'{folder}/{cluster_name}_gmm_metrics_duration_weekhour_{week_hour}.json'
        metrics['grid_search'] = metrics['grid_search'].to_dict(orient="records")
        with open(file_path, 'w') as f:
            json.dump(metrics, f)

    def load_gmm_base_model(self, cluster_name, model_type, day=None):
        """
        Load a base GMM model for a cluster and model type.

        Parameters
        ----------
        cluster_name : str
            Cluster identifier.
        model_type : str
            Type of the model ('start_time', 'energy', 'duration') used in naming.

        Returns
        -------
        object
            The loaded GMM base model.
        """
        if day is not None:
            file_path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/{cluster_name}_gmm_base_model_{model_type}_day_{day}.pkl'
        else:
            file_path = f'{self.model_path}/{self.experiment_name}/generator_models/gmm_models/{cluster_name}_gmm_base_model_{model_type}.pkl'
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model





def filter_out_csv_file_names(data_path, AC_model=False, DC_model=False, timeframe=None):

    csv_file_names = os.listdir(f'{data_path}/power_demands_analysis/{timeframe}')
    csv_file_names = [name for name in csv_file_names if ('acndata' not in name and 'CARL' not in name and 'predict' not in name and 'Korea' not in name and 'China' not in name and 'Asensio2020' not in name)]
    ids = [f.split('_')[0] for f in csv_file_names]
    duplicates = {item for item, count in Counter(ids).items() if count > 1}
    duplicate_files = [f for f in csv_file_names if f.split('_')[0] in duplicates]
    if AC_model:
        csv_file_names = (set(csv_file_names) - set(duplicate_files)).union(f for f in duplicate_files if '_AC_' in f)
        csv_file_names = [f for f in csv_file_names if 'DC' not in f]
    elif DC_model:
        csv_file_names = (set(csv_file_names) - set(duplicate_files)).union(f for f in duplicate_files if '_DC_' in f)
        csv_file_names = [f for f in csv_file_names if 'DC' in f]

    return csv_file_names


            
