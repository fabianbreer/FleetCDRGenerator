import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm, truncnorm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings(
    "ignore",
    category=InconsistentVersionWarning
)


class CDRGenerator:
    """
    Generator for synthetic Charge Detail Records (CDRs).

        This class wraps procedures to fit Gaussian Mixture Models (GMMs) on
        historical CDR data and to sample synthetic charging sessions from those
        models. The class expects a `dataloader` object exposing methods used to
        load and save models and scalers. Required dataloader methods (examples):

        - `load_original_cdrs(cluster, is_AC_model, is_DC_model)` -> DataFrame-like
        - `load_day_probabilities(cluster)` / `save_day_probabilities(...)`
        - `load_gmm_model(cluster, day)` / `save_gmm_model(...)`
        - `load_gmm_model_energy(cluster, week_15_min)` / `save_gmm_model_energy(...)`
        - `load_energy_scaler(...)` / `save_energy_scaler(...)`
        - base variants: `load_gmm_base_model`, `save_gmm_base_model`,
            `load_energy_scaler_base`, `save_energy_scaler_base`, etc.

        Notes
        -----
        - Public methods: `generate_cdrs`, `fit_base_model`, `fit_probabilistic_model`.
            Please note that the fitting methods are not executable as we cannot share the underlying data.
        """
    def __init__(self, dataloader, cluster, is_AC_model=None, is_DC_model=None):
        """
        Initialize the CDR (Charge Detail Record) Generator.

        Parameters
        ----------
        dataloader : Dataloader
            Dataloader instance for loading GMM models and data transformations.
        cluster : str
            Cluster identifier used to load specific model data.
        is_AC_model : bool, optional
            Flag indicating if this is an AC charging model. Default is None.
        is_DC_model : bool, optional
            Flag indicating if this is a DC charging model. Default is None.
        """
        self.dataloader = dataloader
        self.cluster = cluster
        self.is_AC_model = is_AC_model
        self.is_DC_model = is_DC_model
        self.cdr_data = []


    def generate_cdrs(self, method, n_cdrs_per_week=1000, n_weeks=1):
        """
        Generate synthetic CDRs using the specified method.

        Parameters
        ----------
        method : str
            Method to use for CDR generation. Options are 'base' or 'gmm'.
        n_cdrs_per_week : int, optional
            Number of CDRs to generate per week. Default is 1000.
        n_weeks : int, optional
            Number of weeks to generate CDRs for. Default is 1.

        Returns
        -------
        pd.DataFrame
            DataFrame containing generated CDRs.
        """
        if method == 'base':
            self.cdr_data = self._generate_from_base_model(n_cdrs_per_week=n_cdrs_per_week, n_weeks=n_weeks)
            self._post_process_cdrs()
        elif method == 'gmm':
            self.cdr_data = self._generate_from_gmm(n_cdrs_per_week=n_cdrs_per_week, n_weeks=n_weeks)
            self._post_process_cdrs()
        else:
            raise ValueError(f"Unsupported generation method: {method}")
        
        return self.cdr_data
   
    def _post_process_cdrs(self):
        """
        Post-process generated CDRs by converting data types and validating.

        This method performs the following operations:
        - Converts quantity_in_wh, duration_sec, max_socket_power to integers
        - Rounds start_time and end_time to the nearest minute
        - Reorders columns to standard CDR format
        - Validates data integrity (positive durations, valid time ranges, positive energy)
        - Removes records with zero or negative energy

        Raises
        ------
        AssertionError
            If validation checks fail (no data, invalid durations, invalid times).
        """
        self.cdr_data['quantity_in_wh'] = self.cdr_data['quantity_in_wh'].astype(int)
        self.cdr_data['duration_sec'] = self.cdr_data['duration_sec'].astype(int)
        # self.cdr_data['duration_sec'] = max(self.cdr_data['duration_sec'], self.cdr_data['quantity_in_wh']/100000*3600)
        self.cdr_data['max_socket_power'] = self.cdr_data['max_socket_power'].astype(int)
        self.cdr_data['start_time'] = self.cdr_data['start_time'].dt.round('min')
        self.cdr_data['end_time'] = self.cdr_data['end_time'].dt.round('min')
        self.cdr_data['week_15_min'] = self.cdr_data['week_15_min'].astype(int)
        self.cdr_data = self.cdr_data[['cdr_id', 'start_time', 'end_time', 'quantity_in_wh', 'duration_sec', 'max_socket_power']]

        assert len(self.cdr_data) > 0, "No CDR data generated. Please check the generation method."
        assert (self.cdr_data['duration_sec'] >= 0).all(), "Duration in seconds must be positive."
        assert (self.cdr_data['start_time'] <= self.cdr_data['end_time']).all(), "Start time must be before end time."
        invalid_rows = self.cdr_data[self.cdr_data['start_time'] >= self.cdr_data['end_time']]
        if not invalid_rows.empty:
            print("Invalid rows where start_time is not less than end_time:")
            print(invalid_rows)
        self.cdr_data= self.cdr_data[self.cdr_data['quantity_in_wh']>0]
        assert (self.cdr_data['quantity_in_wh'] > 0).all(), "Quantity in Wh must be positive."
    
    def _generate_from_gmm(self, n_cdrs_per_week=20, n_weeks=1):
        """
        Generate CDRs from Gaussian Mixture Models.

        This method samples charging sessions using GMM models trained on historical data.
        It generates start times per day of week, then samples energy and duration values
        from time-specific GMM models with truncation constraints.

        Parameters
        ----------
        n_cdrs_per_week : int, optional
            Number of CDRs to generate per week. Default is 20.
        n_weeks : int, optional
            Number of weeks to simulate. Default is 1.

        Returns
        -------
        pd.DataFrame
            DataFrame containing raw sampled CDRs with columns: cdr_id, week_15_min,
            duration_sec, quantity_in_wh, start_time, end_time, max_socket_power.
        """
        day_probabilities = self.dataloader.load_day_probabilities(self.cluster)
        sampled_cdrs_all_weeks = pd.DataFrame()
        for week in range(n_weeks):
            n_day_samples = np.random.choice(day_probabilities.day.values.astype(int), size=n_cdrs_per_week, p=day_probabilities.probability.values)
            n_day_samples = pd.Series(n_day_samples).value_counts().sort_index()
            n_day_samples = n_day_samples.reindex(range(7), fill_value=0)
            sampled_week_15_min_list = np.array([])
            for day in range(7):
                n = n_day_samples.loc[day]
                if n == 0:
                    continue
                else:
                    # sample start times for the day
                    gmm_model = self.dataloader.load_gmm_model(self.cluster, day)
                    sampled_day_15_min = gmm_model.sample(n)[0].flatten()
                    sampled_day_15_bin = np.mod(np.rint(sampled_day_15_min).astype(int), 96)  # int 0..95
                    sampled_week_15_min = day * 96 + sampled_day_15_bin                  # int 0..671
                    sampled_week_15_min_list = np.concatenate([sampled_week_15_min_list, sampled_week_15_min])

            sampled_duration_sec_list = []
            sampled_quantity_in_wh_list = []
            for sample_week_15_min in tqdm(sampled_week_15_min_list.reshape(-1), desc="Sampling CDRs"):
                energy_scaler = self.dataloader.load_energy_scaler(self.cluster, sample_week_15_min)
                gmm_model_quantity_in_wh = self.dataloader.load_gmm_model_energy(self.cluster, sample_week_15_min)
                bundle_energy = {'gmm': gmm_model_quantity_in_wh, 'scaler': energy_scaler}

                duration_scaler = self.dataloader.load_duration_scaler(self.cluster, sample_week_15_min)
                gmm_model_duration = self.dataloader.load_gmm_model_duration_parallel(self.cluster, sample_week_15_min)
                bundle_duration = {'gmm': gmm_model_duration, 'scaler': duration_scaler}

                Pmax = 22000.0 if self.is_AC_model else 120000.0
                qs = self._sample_univariate_log1p_truncated_one(bundle_energy, lower=1.0, upper=150000.0)
                a_arr = np.maximum(60.0, 3600.0 * qs / Pmax)
                ds = self._sample_univariate_log1p_truncated_one(bundle_duration, lower=a_arr, upper=259200.0)

                sampled_duration_sec_list.append(ds)
                sampled_quantity_in_wh_list.append(qs)

            # Create a DataFrame for the sampled data
            sampled_cdrs = pd.DataFrame({
                'week_15_min': sampled_week_15_min_list.tolist(),
                'duration_sec': sampled_duration_sec_list,
                'quantity_in_wh': sampled_quantity_in_wh_list
            })

            # Map week_15_min back to start_time
            sampled_cdrs['start_time'] = pd.to_datetime('2024-01-01') + pd.Timedelta(weeks=week) + pd.to_timedelta(sampled_cdrs['week_15_min'] * 15, unit='m')

            # Add additional columns based on the sampled data
            sampled_cdrs['end_time'] = sampled_cdrs['start_time'] + pd.to_timedelta(sampled_cdrs['duration_sec'], unit='s')
            sampled_cdrs['max_socket_power'] = sampled_cdrs.apply(
                lambda row: 22 if row['quantity_in_wh'] < row['duration_sec']/3600 * 22000 
                else int(np.clip(round(row['quantity_in_wh'] *3600 / row['duration_sec'] / 10000) * 10, 22, 350)), axis=1
            )
            sampled_cdrs_all_weeks = pd.concat([sampled_cdrs_all_weeks, sampled_cdrs], ignore_index=True)
        sampled_cdrs_all_weeks['cdr_id'] = sampled_cdrs_all_weeks.index+1
        return sampled_cdrs_all_weeks

    def _generate_from_base_model(self, n_cdrs_per_week=20, n_weeks=1):
        """
        Generate CDRs using the "base" GMM models.

        This method mirrors `_generate_from_gmm` but does not employ dependent sampling. 
        It therefore uses different dataloader methods (e.g. `load_gmm_base_model`, 
        `load_energy_scaler_base`, etc.)
        to sample start times, energy and duration values and construct a
        DataFrame of sampled CDR records.

        Parameters
        ----------
        n_cdrs_per_week : int, optional
            Number of CDRs to generate per week. Default is 20.
        n_weeks : int, optional
            Number of weeks to simulate. Default is 1.

        Returns
        -------
        pd.DataFrame
            DataFrame with sampled CDRs (columns include: cdr_id, week_15_min,
            duration_sec, quantity_in_wh, start_time, end_time, max_socket_power).
        """

        day_probabilities = self.dataloader.load_day_probabilities(self.cluster)
        sampled_cdrs_all_weeks = pd.DataFrame()
        for week in range(n_weeks):

            n_day_samples = np.random.choice(day_probabilities.day.values.astype(int), size=n_cdrs_per_week, p=day_probabilities.probability.values)
            n_day_samples = pd.Series(n_day_samples).value_counts().sort_index()
            n_day_samples = n_day_samples.reindex(range(7), fill_value=0)
            sampled_week_15_min_list = np.array([])
            for day in range(7):
                cdr_list = []
                n = n_day_samples.loc[day]
                if n == 0:
                    continue
                # 1) sample start times first (use same binning as gmm_new to be consistent)
                start_model = self.dataloader.load_gmm_base_model(self.cluster, "start_time", day=day)
                sampled_day_15 = start_model.sample(n)[0].flatten()
                sampled_day_15_bin = np.mod(np.rint(sampled_day_15).astype(int), 96)
                sampled_week_15_min = (week * 96) + sampled_day_15_bin 
                sampled_week_15_min_list = np.concatenate([sampled_week_15_min_list, sampled_week_15_min])

            sampled_duration_sec_list = []
            sampled_quantity_in_wh_list = []
            for sample_week_15_min in tqdm(sampled_week_15_min_list.reshape(-1), desc="Sampling CDRs"):
                energy_scaler = self.dataloader.load_energy_scaler_base(self.cluster, day=int(sample_week_15_min//96))
                gmm_model_quantity_in_wh = self.dataloader.load_gmm_base_model(self.cluster, "quantity_in_wh", day=int(sample_week_15_min//96))
                bundle_energy = {'gmm': gmm_model_quantity_in_wh, 'scaler': energy_scaler}

                duration_scaler = self.dataloader.load_duration_scaler_base(self.cluster, day=int(sample_week_15_min//96))
                gmm_model_duration = self.dataloader.load_gmm_base_model(self.cluster, "duration_sec", day=int(sample_week_15_min//96))
                bundle_duration = {'gmm': gmm_model_duration, 'scaler': duration_scaler}

                Pmax = 22000.0 if self.is_AC_model else 120000.0
                qs = self._sample_univariate_log1p_truncated_one(bundle_energy, lower=1.0, upper=150000.0)
                a_arr = np.maximum(60.0, 3600.0 * qs / Pmax)
                ds = self._sample_univariate_log1p_truncated_one(bundle_duration, lower=a_arr, upper=259200.0)

                sampled_duration_sec_list.append(ds)
                sampled_quantity_in_wh_list.append(qs)

            # Create a DataFrame for the sampled data
            sampled_cdrs = pd.DataFrame({
                'week_15_min': sampled_week_15_min_list.tolist(),
                'duration_sec': sampled_duration_sec_list,
                'quantity_in_wh': sampled_quantity_in_wh_list
            })

            # Map week_15_min back to start_time
            sampled_cdrs['start_time'] = pd.to_datetime('2024-01-01') + pd.Timedelta(weeks=week) + pd.to_timedelta(sampled_cdrs['week_15_min'] * 15, unit='m')

            # Add additional columns based on the sampled data
            sampled_cdrs['end_time'] = sampled_cdrs['start_time'] + pd.to_timedelta(sampled_cdrs['duration_sec'], unit='s')
            sampled_cdrs['max_socket_power'] = sampled_cdrs.apply(
                lambda row: 22 if row['quantity_in_wh'] < row['duration_sec']/3600 * 22000 
                else int(np.clip(round(row['quantity_in_wh'] *3600 / row['duration_sec'] / 10000) * 10, 22, 350)), axis=1
            )
            sampled_cdrs_all_weeks = pd.concat([sampled_cdrs_all_weeks, sampled_cdrs], ignore_index=True)
        sampled_cdrs_all_weeks['cdr_id'] = sampled_cdrs_all_weeks.index+1
        return sampled_cdrs_all_weeks

    def _mu_sigma_1d(self, gmm):
        """
        Extract 1D GMM parameters: means and standard deviations per component.

        Parameters
        ----------
        gmm : sklearn.mixture.GaussianMixture
            Fitted 1D Gaussian Mixture Model.

        Returns
        -------
        mu : np.ndarray
            Mean values for each component, shape (K,).
        sigma : np.ndarray
            Standard deviations for each component, shape (K,).
        weights : np.ndarray
            Component weights (mixing coefficients), shape (K,).

        Raises
        ------
        ValueError
            If the covariance type is unsupported.
        """
        K = gmm.n_components
        mu = gmm.means_.ravel()
        ct = gmm.covariance_type
        cov = gmm.covariances_
        if ct == "full":       sigma = np.sqrt(cov[:, 0, 0])
        elif ct == "diag":     sigma = np.sqrt(cov[:, 0])       # shape (K, 1) -> (K,)
        elif ct == "tied":     sigma = np.sqrt(cov[0, 0]) * np.ones(K)
        elif ct == "spherical":sigma = np.sqrt(cov)             # shape (K,)
        else:
            raise ValueError(f"Unsupported covariance_type: {ct}")
        # Numerical stability
        sigma = np.maximum(sigma, 1e-8)
        return mu, sigma, gmm.weights_.ravel()

    def _bounds_to_z(self, scaler, lower, upper):
        """
        Convert original bounds to Z-bounds (standardized log-space).

        Transforms bounds from original space through log1p and StandardScaler
        transformation to standardized Z-space used by the GMM models.

        Parameters
        ----------
        scaler : sklearn.preprocessing.StandardScaler
            Fitted StandardScaler instance used for log-transformed data.
        lower : float or array-like
            Lower bound(s) in original units. Can be scalar or array.
        upper : float or array-like
            Upper bound(s) in original units. Can be scalar or array.

        Returns
        -------
        a : np.ndarray
            Lower bound(s) in standardized Z-space.
        b : np.ndarray
            Upper bound(s) in standardized Z-space.
        """
        lower = np.asarray(lower, dtype=float).ravel()
        upper = np.asarray(upper, dtype=float).ravel()
        assert lower.shape == upper.shape
        n = lower.size

        a = np.empty(n, dtype=float)
        b = np.empty(n, dtype=float)

        # Masks for finite bounds
        mL = np.isfinite(lower)
        mU = np.isfinite(upper)

        # Log1p, then StandardScaler affine transformation
        if np.any(mL):
            yL = np.log1p(lower[mL]).reshape(-1, 1)
            a[mL] = (yL[:, 0] - scaler.mean_[0]) / scaler.scale_[0]
        a[~mL] = -np.inf

        if np.any(mU):
            yU = np.log1p(upper[mU]).reshape(-1, 1)
            b[mU] = (yU[:, 0] - scaler.mean_[0]) / scaler.scale_[0]
        b[~mU] = np.inf

        return a, b

    def _sample_univariate_log1p_truncated_one(self, bundle, lower, upper):
        """
        Sample one value from truncated 1D-GMM in log1p+StandardScaler space.

        Performs exact truncated sampling from a 1D Gaussian Mixture Model that was
        trained on log1p-transformed and standardized data. The truncation bounds
        are specified in original units and automatically transformed.

        Parameters
        ----------
        bundle : dict
            Dictionary containing 'gmm' (fitted GaussianMixture) and 'scaler'
            (fitted StandardScaler) used for the transformation.
        lower : float
            Lower truncation bound in original units (e.g., seconds or Wh).
        upper : float
            Upper truncation bound in original units (e.g., seconds or Wh).

        Returns
        -------
        float or None
            Sampled value in original space, or None if the truncation region
            is invalid or has negligible probability mass.
        """
        gmm = bundle["gmm"]; scaler = bundle["scaler"]
        mu, sigma, pi = self._mu_sigma_1d(gmm)

        # Bounds in Z-space
        a, b = self._bounds_to_z(scaler, lower=[lower], upper=[upper])
        a = a[0]; b = b[0]
        if not (a < b):
            return None  # invalid range

        # Truncated masses per component
        alpha = (a - mu) / sigma
        beta  = (b - mu) / sigma
        r     = norm.cdf(beta) - norm.cdf(alpha)        # Component mass in the [a,b] interval

        mass = float(np.dot(pi, r))
        if mass <= 1e-12:
            return None  # no mass in the allowed range

        w = (pi * r) / mass                             # renormalized weights
        k = np.random.choice(len(w), p=w)

        # Sample from the truncated normal of the selected component
        z = truncnorm.rvs(alpha[k], beta[k], loc=mu[k], scale=sigma[k])
        # back to original space
        y = scaler.inverse_transform([[z]])[0, 0]
        x = np.expm1(y)
        return x

    def fit_base_model(self, max_components = 10):
        """
        Fit and save "base" GMM models and scalers for start time, energy, and duration.

        This routine loads original CDRs from the dataloader, computes simple
        time-derived features and fits GaussianMixture models (using BIC via
        GridSearchCV) and StandardScaler instances. Results are saved using the
        dataloader's `save_*_base` methods.

        Parameters
        ----------
        max_components : int, optional
            Maximum number of mixture components to try when selecting GMMs
            (default 10).
        """
        data = self.dataloader.load_original_cdrs(self.cluster, self.is_AC_model, self.is_DC_model)
        cdrs = pd.DataFrame()
        cdrs['start_time'] = pd.to_datetime(data['start_time'])
        cdrs['start_time'] = pd.to_datetime(cdrs['start_time'])
        cdrs['end_time_idle'] = pd.to_datetime(data['end_time_idle'])
        cdrs['end_time'] = pd.to_datetime(data['end_time'])
        cdrs['week_hour'] = ((cdrs['start_time'].dt.weekday * 24) + cdrs['start_time'].dt.hour)
        cdrs['week_hour_end'] = ((cdrs['end_time_idle'].dt.weekday * 24) + cdrs['end_time_idle'].dt.hour)
        cdrs['week_15_min'] = ((cdrs['start_time'].dt.weekday * 24 * 4) + (cdrs['start_time'].dt.hour * 4) + (cdrs['start_time'].dt.minute // 15))
        cdrs['quantity_in_wh'] = data['quantity_in_wh']
        cdrs['duration_sec'] = data['duration_sec']
        cdrs['day_15_min'] = (cdrs['start_time'].dt.hour * 4) + (cdrs['start_time'].dt.minute // 15)
        
        day_probabilities = cdrs['start_time'].dt.weekday.value_counts(normalize=True).sort_index()
        self.dataloader.save_day_probabilities(day_probabilities, self.cluster)
        for day in tqdm(range(7), desc=f"Fitting GMM for cluster {self.cluster} Start Time"):
            grid_search = self._define_gmm_param_grid(max_components)
            cdrs_day = cdrs[cdrs['start_time'].dt.weekday == day]
            
            # fitting for start times
            cdrs_day = cdrs[cdrs['start_time'].dt.weekday == day]
            # Fit a Gaussian Mixture Model to the week_15_min data
            X = cdrs_day[['day_15_min']].values.astype(float)
            grid_search.fit(X)
            self.dataloader.save_gmm_base_model(grid_search.best_estimator_, self.cluster, 'start_time', day=day)

            # fitting for energy
            grid_search = self._define_gmm_param_grid(max_components)
            scaler = StandardScaler()
            X = cdrs_day[['quantity_in_wh']]
            X_log = np.log1p(X)
            X_scaled = scaler.fit_transform(X_log)
            grid_search.fit(X_scaled)
            self.dataloader.save_energy_scaler_base(scaler, self.cluster, day)
            self.dataloader.save_gmm_base_model(grid_search.best_estimator_, self.cluster, 'quantity_in_wh', day=day)

            # fitting for duration
            grid_search = self._define_gmm_param_grid(max_components)
            scaler = StandardScaler()
            X = cdrs_day[['duration_sec']]
            X_log = np.log1p(X)
            X_scaled = scaler.fit_transform(X_log)
            grid_search.fit(X_scaled)
            self.dataloader.save_duration_scaler_base(scaler, self.cluster, day)
            self.dataloader.save_gmm_base_model(grid_search.best_estimator_, self.cluster, 'duration_sec', day=day)

    def fit_probabilistic_model(self, max_components_step_1=10, max_components_step_2=10, max_components_step_3=10):
        """
        Fit probabilistic GMM models for start time, energy and duration.

        The fitting proceeds in stages: first a per-day GMM for start times
        (coarser), then per-week-hour GMMs/scalers for energy and duration
        (finer resolution). Selected models and scalers are saved via the
        dataloader (metrics are no longer saved).

        Parameters
        ----------
        max_components_step_1 : int, optional
            Maximum components for start-time GMM fitting (default 10).
        max_components_step_2 : int, optional
            Maximum components for energy GMM fitting (default 10).
        max_components_step_3 : int, optional
            Maximum components for duration GMM fitting (default 10).
        """
        data = self.dataloader.load_original_cdrs(self.cluster, self.is_AC_model, self.is_DC_model)
        cdrs = pd.DataFrame()
        cdrs['start_time'] = pd.to_datetime(data['start_time'])
        cdrs['start_time'] = pd.to_datetime(cdrs['start_time'])
        cdrs['end_time_idle'] = pd.to_datetime(data['end_time_idle'])
        cdrs['end_time'] = pd.to_datetime(data['end_time'])
        cdrs['week_hour'] = ((cdrs['start_time'].dt.weekday * 24) + cdrs['start_time'].dt.hour)
        cdrs['week_hour_end'] = ((cdrs['end_time_idle'].dt.weekday * 24) + cdrs['end_time_idle'].dt.hour)
        cdrs['week_15_min'] = ((cdrs['start_time'].dt.weekday * 24 * 4) + (cdrs['start_time'].dt.hour * 4) + (cdrs['start_time'].dt.minute // 15)) 
        cdrs['quantity_in_wh'] = data['quantity_in_wh']
        cdrs['duration_sec'] = data['duration_sec']
        cdrs['day_15_min'] = (cdrs['start_time'].dt.hour * 4) + (cdrs['start_time'].dt.minute // 15)
        
        # ----------------------------------------------------------------------------------------------------------
        # fit GMM for start time (week_15_min)
        grid_search = grid_search = self._define_gmm_param_grid(max_components_step_1)
        day_probabilities = cdrs['start_time'].dt.weekday.value_counts(normalize=True).sort_index()
        self.dataloader.save_day_probabilities(day_probabilities, self.cluster)
        for day in tqdm(range(7), desc=f"Fitting GMM for cluster {self.cluster} Start Time"): 
            cdrs_day = cdrs[cdrs['start_time'].dt.weekday == day]
            # Fit a Gaussian Mixture Model to the week_15_min data
            X = cdrs_day[['day_15_min']].values.astype(float)
            grid_search.fit(X)
            self.dataloader.save_gmm_model(grid_search.best_estimator_, self.cluster, day)

        # ----------------------------------------------------------------------------------------------------------
        # fit GMM for energy and duration
        for week_hour in tqdm(range(168), desc="Fitting GMM for Energy and Duration"):
            
            # energy
            grid_search = grid_search = self._define_gmm_param_grid(max_components_step_2)
            
            is_fitted = False
            counter = 1
            while not is_fitted:
                cdrs_week = cdrs[(cdrs['week_hour'] >= week_hour-counter) & (cdrs['week_hour'] <= week_hour+counter)]
                try:
                    scaler = StandardScaler()
                    X = cdrs_week[['quantity_in_wh']]
                    X_log = np.log1p(X)
                    X_scaled = scaler.fit_transform(X_log)
                    grid_search.fit(X_scaled)
                    self.dataloader.save_energy_scaler(scaler, self.cluster, week_hour)
                    self.dataloader.save_gmm_model_energy(grid_search.best_estimator_, self.cluster, week_hour)
                    is_fitted = True
                except Exception as e:
                    counter += 1
            
            # duration
            grid_search = self._define_gmm_param_grid(max_components_step_3)
            is_fitted = False
            counter = 1
            while not is_fitted:
                cdrs_week = cdrs[(cdrs['week_hour'] >= week_hour-counter) & (cdrs['week_hour'] <= week_hour+counter)]
                try:
                    scaler = StandardScaler()
                    X = cdrs_week[['duration_sec']]
                    X_log = np.log1p(X)
                    X_scaled = scaler.fit_transform(X_log)
                    grid_search.fit(X_scaled)
                    self.dataloader.save_duration_scaler(scaler, self.cluster, week_hour)
                    self.dataloader.save_gmm_model_duration_parallel(grid_search.best_estimator_, self.cluster, week_hour)
                    is_fitted = True
                except Exception as e:
                    counter += 1

    def _gmm_bic_score(self, estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)
    
    def _define_gmm_param_grid(self, max_components): 
        param_grid = {
            "n_components": range(1, max_components + 1),
            "covariance_type": ["diag"],
            "n_init": [5],
            "random_state": [42]
        }
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=self._gmm_bic_score
        )
        return grid_search
