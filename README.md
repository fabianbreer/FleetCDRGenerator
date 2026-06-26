# FleetCdrSynthesis

A research tool for hierarchical clustering and synthetic generation of electric vehicle charge detail records (CDRs). This package accompanies the journal paper [Paper Title/Citation - To be added].

## Repository organization

For ease of use this repository has been cleaned to include only the files required to run the CDR generator and clustering tools. The main folders and files you will interact with are:

- `src/` — core Python modules: `cdr_generator.py`, `cluster_tree_model.py`, `dataloader.py`, and helpers.
- `scripts/` — convenience scripts for training, model management and running generation pipelines.
- `data/` — example and preprocessed data used by the generator and clustering routines.
- `models/` and `results/` — locations where trained models and output artifacts are saved by the scripts.
- `config.json` — top-level configuration used by the dataloader and scripts.
- `requirements.txt` — Python dependencies for the cleaned workspace.

If you need the original, broader project history (figures, auxiliary analyses and the full artifact set used for the journal paper), please switch to the `journal-paper` branch which contains the complete repository state and supplementary materials.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Module Documentation](#module-documentation)
  - [CDRGenerator](#cdrgenerator)
  - [ClusterTreeModel](#clustertreemodel)
  - [Dataloader](#dataloader)
  - [Data Process](#data-process)
- [Citation](#citation)

## Installation

To set up the environment, create a virtual environment and install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Running scripts

Scripts in `scripts/` are intended to be executed from the project root. The
top-level scripts append `src/` to `sys.path` so imports like `from dataloader import *`
work when running the script directly. Example (run from repository root):

```bash
python scripts/synthesize_cdrs.py
```

If you prefer to import modules directly in your own code, you can either add
the repository `src/` directory to `PYTHONPATH` or import explicitly as a
package, for example `from src.dataloader import Dataloader` when running code
from outside the project tree.

### Cluster labels and internal IDs

Human-readable labels map to internal cluster IDs (defined in `config.json`).
Example mapping:

| Human label AC | Internal ID |
|---|---:|
| A1 | 00010 |
| A2 | 00011 |
| A3 | 00101 |
| A4 | 00110 |
| B1 | 100 |
| B2 | 1011 |
| B3 | 11 |

| Human label DC | Internal ID |
|---|---:|
| C1 | 11 |
| D1 | 0100 |
| D2 | 01013 |
| D3 | 011 |

Translate programmatically with `Dataloader`:

```python
internal_id = dataloader.cluster_id_from_label('A1', model='AC')
label = dataloader.cluster_label('00010', model='AC')
```

Configure the generation parameters by editing the `config.json` file in the project root directory:

- **`cluster`**: Cluster identifier (e.g., 'A1', 'C1')
- **`n_cdrs_per_weeks`**: Number of CDRs to generate per week
- **`n_weeks`**: Number of weeks to simulate
- **`method`**: Generation method selection

---

## Module Documentation

This section provides technical documentation for the main classes in the FleetCdrSynthesis package.

### CDRGenerator

Class for generating synthetic charge detail records (CDRs) using Gaussian Mixture Models.

**Location:** `src/cdr_generator.py`

#### Public Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `dataloader` (Dataloader)<br>`cluster` (str)<br>`is_AC_model` (bool, optional)<br>`is_DC_model` (bool, optional) | None | Initialize the CDR Generator with a dataloader instance and cluster configuration. |
| `generate_cdrs` | `n_cdrs_per_week` (int, default=1000)<br>`n_weeks` (int, default=1) | pd.DataFrame | Generate synthetic CDRs with columns: cdr_id, start_time, end_time, quantity_in_wh, duration_sec, max_socket_power. |
| `post_process_cdrs` | None | None | Post-process generated CDRs by converting data types, validating integrity, and removing invalid records. |

#### Output Format

The `generate_cdrs()` method returns a pandas DataFrame with the following columns:

| Column | Data Type | Unit | Description |
|--------|-----------|------|-------------|
| `cdr_id` | int | - | Sequential identifier for each charging session (1, 2, 3, ...) |
| `start_time` | datetime64 | - | Start timestamp of the charging session, rounded to the nearest minute |
| `end_time` | datetime64 | - | End timestamp of the charging session, rounded to the nearest minute |
| `quantity_in_wh` | int | Wh | Total energy delivered during the charging session in watt-hours |
| `duration_sec` | int | seconds | Duration of the charging session in seconds |
| `max_socket_power` | int | kW | Maximum charging power of the socket, calculated from energy and duration (range: 22-350 kW, rounded to nearest 10 kW) |

The `start_time`, `duration_sec`, and `quantity_in_wh` are direct results from the trained model. The `max_socket_power` column indicates the minimum socket power required to ensure feasibility of the charge detail record. The minimum value for this column is 22 kW, as most publicly available slow charging chargepoints offer this power.

**Output Constraints:**
- All timestamps are rounded to the nearest minute
- `duration_sec` must be positive (> 0)
- `quantity_in_wh` must be positive (> 0)
- `start_time` must be before or equal to `end_time`
- Records with invalid values are automatically removed during post-processing

**Example Output:**
```
   cdr_id          start_time            end_time  quantity_in_wh  duration_sec  max_socket_power
0       1 2024-01-01 08:15:00 2024-01-01 09:30:00           15420          4500                22
1       2 2024-01-01 14:23:00 2024-01-01 16:45:00           28950          8520                22
2       3 2024-01-02 07:08:00 2024-01-02 08:12:00           12100          3840                22
```

**Usage Example:**

```python
from src.dataloader import Dataloader
from src.cdr_generator import CDRGenerator

dataloader = Dataloader(base_path=".", experiment_name="AC_model_gmm")
generator = CDRGenerator(dataloader, cluster='100', is_AC_model=True)
cdrs = generator.generate_cdrs(n_cdrs_per_week=30, n_weeks=2)
```

---

### ClusterTreeModel

Class for hierarchical time series clustering of electric vehicle fleet charging patterns using Dynamic Time Warping (DTW).

**Location:** `src/cluster_tree_model.py`

#### Public Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `dataloader` (Dataloader)<br>`AC_model` (bool, default=True)<br>`DC_model` (bool, default=False)<br>`experiment_name` (str, optional)<br>`threshold_size` (int, default=21) | None | Initialize the Cluster Tree Model for hierarchical time series clustering. |
| `set_seed` | `seed` (int) | None | Set random seed for reproducibility across clustering operations. |
| `train` | None | None | Train the hierarchical cluster tree model. Performs two-stage clustering: initial K-means on weekday/weekend features, then recursive DTW-based clustering. |
| `save_model` | `filename` (str) | None | Save the trained cluster tree model to disk. |
| `predict` | `fleet_id` (str) | str | Predict the cluster assignment for a fleet by hierarchically traversing the tree. Returns cluster identifier. |

**Usage Example:**

```python
from src.dataloader import Dataloader
from src.cluster_tree_model import ClusterTreeModel

dataloader = Dataloader(base_path=".", experiment_name="AC_model_gmm")
model = ClusterTreeModel(dataloader, AC_model=True, threshold_size=21)
model.train()
model.save_model("cluster_model.pkl")

# Predict cluster for a fleet
cluster = model.predict(fleet_id="12345")
```

---

### Dataloader

Class for centralized data and model I/O operations. Manages loading/saving of models, scalers, and preprocessed data.

**Location:** `src/dataloader.py`

#### Public Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `base_path` (str)<br>`experiment_name` (str, optional) | None | Initialize the Dataloader with repository root path and experiment name. Creates necessary directory structure. |
| `save_model` | `model` (object)<br>`subfolder` (str)<br>`filename` (str) | None | Save a Python object (model) to the experiment model folder using pickle. |
| `load_model` | `subfolder` (str)<br>`filename` (str) | object | Load a pickled model object from the experiment model folder. |
| `save_submodel` | `model` (object)<br>`subfolder` (str)<br>`filename` (str) | None | Save a model into the submodels directory under an experiment subfolder. |
| `load_ts_submodel` | `subfolder` (str)<br>`cluster_name` (str) | object | Load a time-series submodel for a cluster. Filename constructed as `{cluster_name}_model.pkl`. |
| `load_submodel` | `subfolder` (str)<br>`cluster_name` (str) | object | Load a submodel saved under submodels directory for a given cluster name. |
| `get_all_models` | `subfolder` (str) | list of str | List available submodel prefixes present in a subfolder's submodels directory. |
| `get_in_array` | `timeframe` (str)<br>`AC_model` (bool, default=True)<br>`DC_model` (bool, default=False) | tuple | Load precomputed time series arrays for clustering. Returns (input_array, csv_file_names). |
| `get_cluster_map` | None | dict | Return the cluster_map dictionary from config.json mapping cluster IDs to labels. |
| `reverse_cluster_map` | None | dict | Return a reversed cluster map: label → cluster ID for each model (AC/DC). |
| `cluster_label` | `cluster_id` (str)<br>`model` (str, default='AC') | str | Return the human-readable label for a cluster ID (e.g., 'A1'). |
| `cluster_id_from_label` | `label` (str)<br>`model` (str, default='AC') | str | Return the internal cluster ID for a given human-readable label. |
| `save_csv_files` | `clusters` (array-like)<br>`names` (sequence)<br>`ids` (sequence)<br>`variances` (sequence)<br>`distances` (sequence) | None | Save cluster assignment CSVs and update cluster sizes metadata. |
| `save_gmm_model` | `model` (object)<br>`cluster_name` (str)<br>`day` (int) | None | Persist a GMM model used for start-time generation for a cluster/day. |
| `save_gmm_model_energy` | `model` (object)<br>`cluster_name` (str)<br>`week_hour` (int) | None | Save an energy GMM model for a cluster at a given week_hour bin. |
| `save_energy_scaler` | `scaler` (object)<br>`cluster_name` (str)<br>`week_hour` (int) | None | Persist an energy scaler object for a cluster/week_hour. |
| `save_duration_scaler` | `scaler` (object)<br>`cluster_name` (str)<br>`week_hour` (int) | None | Persist a duration scaler object for a cluster/week_hour. |
| `load_energy_scaler` | `cluster_name` (str)<br>`week_hour` (int) | object | Load a previously saved energy scaler for a cluster/week_hour. |
| `load_duration_scaler` | `cluster_name` (str)<br>`week_hour` (int) | object | Load a previously saved duration scaler for a cluster/week_hour. |
| `load_gmm_model_energy` | `cluster_name` (str)<br>`week_hour` (int) | object | Load a saved energy GMM model for a cluster at a particular week_hour. |
| `save_gmm_model_duration` | `model` (object)<br>`cluster_name` (str)<br>`week_hour` (int)<br>`energy_bin` (int) | None | Save a duration GMM model for a cluster/week_hour/energy_bin. |
| `load_gmm_model_duration` | `cluster_name` (str)<br>`week_hour` (int)<br>`quantity_in_wh` (numeric) | object | Load a duration GMM model for a cluster, selecting an energy bin based on quantity. |
| `load_gmm_model_duration_parallel` | `cluster_name` (str)<br>`week_hour` (int) | object | Load a duration GMM model (parallel version) that was saved per week_hour. |
| `save_gmm_model_duration_parallel` | `model` (object)<br>`cluster_name` (str)<br>`week_hour` (int) | None | Save the parallel duration GMM model for a cluster/week_hour. |
| `load_gmm_model` | `cluster_name` (str)<br>`day` (int) | object | Load a start-time GMM model for a cluster and day index (0-6). |
| `save_day_probabilities` | `day_probabilities` (pd.Series or dict)<br>`cluster_name` (str) | None | Save a pandas Series or mapping of day probabilities for a cluster to JSON. |
| `load_day_probabilities` | `cluster_name` (str) | pd.DataFrame | Load day probabilities JSON for a cluster and return as DataFrame with columns ['day', 'probability']. |
| `load_mean_power_profile` | `cluster` (str) | pd.Series or pd.DataFrame | Retrieve the precomputed mean power profile for a cluster. |
| `load_example_cdrs` | `n_samples` (int, default=1000) | pd.DataFrame | Generate a small synthetic DataFrame of example CDRs for testing/inspection. |

**Usage Example:**

```python
from src.dataloader import Dataloader

# Initialize dataloader
dataloader = Dataloader(base_path=".", experiment_name="my_experiment")

# Load time series data
in_array, filenames = dataloader.get_in_array('weekly_average', AC_model=True)

# Get cluster mapping
cluster_map = dataloader.get_cluster_map()
label = dataloader.cluster_label('00010', model='AC')  # Returns 'A1'

# Load models
gmm_model = dataloader.load_gmm_model(cluster_name='100', day=0)
scaler = dataloader.load_energy_scaler(cluster_name='100', week_hour=24)
```

---


### Data Process

Utilities for cleaning and converting raw CDRs into minute-resolution
power and occupancy time series, and for producing aggregated weekly and
weekday/weekend summaries used by the clustering and generator pipelines.

**Location:** src/data_process.py

#### Highlights

- `find_p_demand_for_each_charging_event_from_charging_data(...)` — preprocess CDRs and construct per-session minute-resolution power curves.
- `generate_p_demand_occupancy_timeseries(...)` — aggregate per-session curves into cluster-level power demand and occupancy time series.
- `process_weekly_data(...)`, `process_weekday_weekend_data(...)` — produce and save weekly and weekday/weekend averaged CSVs under `data/power_demands_analysis/`.
- Helpers for resampling, DC profile scaling (`_use_dc_profiles`), and computing weekly/weekday statistics used across the preprocessing pipeline.

**Usage example:**

```python
from src.data_process import find_p_demand_for_each_charging_event_from_charging_data, generate_p_demand_occupancy_timeseries

processed, session_dict = find_p_demand_for_each_charging_event_from_charging_data(raw_df, base_path=".")
p_demand, processed = generate_p_demand_occupancy_timeseries(processed, session_dict, min_resolution=15)
```

## Citation
If you use this tool in your research, please cite our paper (not yet published):

```bibtex
@article{Breer2026,
  title={[Behavioral Modeling of Corporate Charging Hubs Introducing a Novel Clustering Method]},
  author={[Breer, Fabian; Gong, Jingyu; Junker, Mark; Zhang, Lei; Huang, Zhijia; Sauer, Dirk Uwe]},
  journal={[Energy and AI]},
  year={2026},
  doi={[https://doi.org/10.1016/j.egyai.2026.100817]}
}
```

## License

See [LICENSE](LICENSE) file for details.


