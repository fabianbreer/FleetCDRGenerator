import pandas as pd
import numpy as np
import datetime
from datetime import datetime as dt
from datetime import timedelta
import os
import pytz

"""Helpers for processing charging data and producing power/occupancy
time series and aggregated summaries.

This module contains utilities used across the project to convert
standard DC/AC power curves, resample and align charging time series,
and compute weekly/weekday-weekend aggregates.
"""

WEEKLIST = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
DATANAME = None
MONDAY = dt.strptime("10 10 2022 00:00", "%d %m %Y %H:%M")
WEEKDAY = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']


def find_p_demand_for_each_charging_event_from_charging_data(charging_data:pd.DataFrame, base_path, date_cutoff = datetime.datetime(year=2020, month=1, day=1), end_date_cutoff = datetime.datetime(year=2024, month=6, day=30)) -> pd.DataFrame:
    """
    This sub-function pre-processes the CDRs and make the data ready to find the power demand profile

    Parameters
    ----------
    charging_data : pd.DataFrame
        Unprocessed CDRs

    Returns
    -------
    charging_data : pd.DataFrame
        processed CDRs to extract the power demand profile

    """
    charging_data_dict = {}
    charging_data = charging_data.copy()
    size_before = charging_data.shape
    charging_data = charging_data.drop_duplicates().reset_index(drop=True)
    charging_data = charging_data[charging_data.quantity_in_wh<120*1000].reset_index(drop=True)

    charging_data = charging_data.copy()



    charging_data['start_time'] = _strip_timezone_column(charging_data['start_time'])
    charging_data['end_time'] = _strip_timezone_column(charging_data['end_time'])


    # Delete incorrect data
    date_cutoff = date_cutoff
    end_date_cutoff = end_date_cutoff

    # Filtering of unwanted or infeasible CDRs
    charging_data = charging_data[charging_data['start_time']>=date_cutoff]
    charging_data = charging_data[charging_data['end_time']>=date_cutoff]
    charging_data = charging_data[charging_data['end_time']<end_date_cutoff]
    charging_data = charging_data[charging_data['end_time']!=charging_data['start_time']]
    charging_data = charging_data[charging_data['quantity_in_wh']/1000>0.1]

    if charging_data.empty:
        return pd.DataFrame(), charging_data_dict
    
    charging_data.reset_index(drop=True, inplace=True)

 
    # Check if start time and end time are correct
    for i in range(len(charging_data)):
        if charging_data.loc[charging_data.index[i], 'start_time'] > charging_data.loc[charging_data.index[i], 'end_time']:
            charging_data.loc[charging_data.index[i], 'end_time'], charging_data.loc[charging_data.index[i], 'start_time'] = charging_data.loc[charging_data.index[i], 'start_time'], charging_data.loc[charging_data.index[i], 'end_time']


    # Find total charging time
    charging_data['delta'] = (charging_data['end_time'] - charging_data['start_time'])
    charging_data['duration_sec']= charging_data['delta'].apply(_get_total_seconds)


    drop_indices = []
    # convert total time to hours
    charging_data['totaltime_hour'] = charging_data['delta'].dt.total_seconds() / 3600
    charging_data['totaltime_idle'] = charging_data['totaltime_hour']
    charging_data['end_time_idle'] = charging_data['end_time']

    
    charging_data['fastest_charging_time'] = charging_data.apply(lambda row: datetime.timedelta(minutes=int((row['quantity_in_wh'] * 60) / (row['max_socket_power'] * 1000))), axis=1)

    # create a mean power curve for each fleet
    power_levels = charging_data.max_socket_power.unique()

    # Align with methodology of paper and set all power levels below 11kW to 11kW (since the mean profiles are based on 11kW and 22kW)
    power_levels = [power if power > 11 else 11 for power in power_levels]
    charging_data['max_socket_power'] = charging_data['max_socket_power'].map(lambda x: x if x > 11 else 11)

    for level in power_levels:
        charging_data_level = charging_data[charging_data['max_socket_power']==level]
        for index, row in charging_data_level.iterrows():


            if level==22 or level==11:
                # For AC it doesnt make sense to use mean profiles
                if 11*row.duration_sec/3600>row.quantity_in_wh/1000:
                    charging_time = row.quantity_in_wh/1000*1/11*60
                    charging_curve = np.ones(int(round(charging_time)))*11
                elif 11*row.duration_sec/3600<row.quantity_in_wh/1000 and level>11:
                    charging_time = row.quantity_in_wh/1000*1/22*60
                    charging_curve = np.ones(int(round(charging_time)))*22
                else:
                    charging_time = row.quantity_in_wh/1000*1/11*60
                    charging_curve = np.ones(int(round(charging_time)))*11
            elif level == 20:
                if 11*row.duration_sec/3600>row.quantity_in_wh/1000:
                    charging_time = row.quantity_in_wh/1000*1/11*60
                    charging_curve = np.ones(int(round(charging_time)))*11
                elif 11*row.duration_sec/3600<row.quantity_in_wh/1000 and level>11:
                    charging_time = row.quantity_in_wh/1000*1/20*60
                    charging_curve = np.ones(int(round(charging_time)))*20
                else:
                    charging_time = row.quantity_in_wh/1000*1/11*60
                    charging_curve = np.ones(int(round(charging_time)))*11
            else:
                # only DC
                charging_curve, charging_time = _use_dc_profiles(row, base_path)
                charging_curve = charging_curve['power'].values
                charging_curve = [round(value, 2) for value in charging_curve * row.quantity_in_wh / sum(1000 * charging_curve / 60)]
                daterange = pd.date_range(start=row.start_time, freq='min', periods=len(charging_curve)).values
                assert len(daterange)==len(charging_curve), 'Dimensions of variables daterange and charging_curve are unequal. Dimension daterange: {}. Dimension charging_curve: {}.'.format(len(daterange), len(charging_curve))
                df = pd.DataFrame({'datetime': daterange,'power': charging_curve})
                if df.empty:
                    drop_indices.append(index)
                else:
                    df['measure_value']=np.nan
                    df['transfered_energy'] = np.nan
                    df['time_diff'] = np.nan 
                    charging_data_dict[row.cdr_id] = df
                    assert (df.power<=row.max_socket_power).all()
            daterange = pd.date_range(start=row.start_time, freq='1min', periods=len(charging_curve)).values
            assert len(daterange)==len(charging_curve), 'Dimensions of variables daterange and charging_curve are unequal. Dimension daterange: {}. Dimension charging_curve: {}.'.format(len(daterange), len(charging_curve))
            df = pd.DataFrame({'datetime': daterange,'power': charging_curve})
            if df.empty:
                drop_indices.append(index)
                # charging_data_meter = charging_data_meter[charging_data_meter['cdr_id']!=row.cdr_id]
                # charging_data_meter = charging_data_meter.reset_index(drop=True)
            else:
                df['measure_value']=np.nan
                df['transfered_energy'] = np.nan
                df['time_diff'] = np.nan 
                charging_data_dict[row.cdr_id] = df
                assert (df.power<=row.max_socket_power).all()

    charging_data = charging_data.drop(drop_indices)

    return charging_data, charging_data_dict

def generate_p_demand_occupancy_timeseries(charging_data: pd.DataFrame, charging_data_meter_dict: dict, min_resolution: int, localize_dt:bool=True) -> pd.DataFrame:
    """Generate cluster-level power demand and occupancy time series.

    Parameters
    ----------
    charging_data : pd.DataFrame
        DataFrame of charging records with at least `start_time`, `end_time`,
        `end_time_idle`, `cdr_id`, and `max_socket_power`.
    charging_data_meter_dict : dict
        Mapping from `cdr_id` to minute-resolution DataFrame containing the
        measured `power` and transfer-related fields.
    min_resolution : int
        Time resolution in minutes for the output time series.
    localize_dt : bool
        If True, localize resulting datetime columns to Europe/Berlin timezone.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple of (p_demand_timeseries, charging_data) where the first is the
        timeseries DataFrame and the second is the (possibly localized)
        augmented charging_data input.
    """

    charging_data = charging_data.copy()
    charging_data = charging_data.sort_values(by='start_time')
    charging_data.reset_index(drop=True, inplace=True)
    # Extract the first start_time value
    first_start_time = charging_data.start_time.iloc[0]

    # Get year, month, and day as separate values
    year = first_start_time.year
    month = first_start_time.month
    day = first_start_time.day

    start_time = datetime.datetime(year=year, month=month, day=day, hour=0, minute=0, second=0)
    end_time = max(charging_data['end_time'])
    resolution = datetime.timedelta(minutes=min_resolution)

    # Create a time range using NumPy
    time_range = np.arange(start_time, end_time, resolution)

    result_time = time_range#[:-1]  # Exclude the last interval
    result_power = np.zeros_like(result_time, dtype=float)
    result_occupancy = np.zeros_like(result_time, dtype=float)
    result_occupancy_idle = np.zeros_like(result_time, dtype=float)
    result_occupancy_dc = np.zeros_like(result_time, dtype=float)
    result_occupancy_dc_idle = np.zeros_like(result_time, dtype=float)
    

    def resample_and_align(time_series_dict, master_index, freq, charging_data):
        """Resample per-`cdr_id` time series into a common master index.

        Parameters
        ----------
        time_series_dict : dict
            Mapping of `cdr_id` to minute-resolution DataFrame with columns
            `measure_value`, `transfered_energy`, `time_diff`, and `power`.
        master_index : array-like
            The target datetime index to align to (already constructed by
            the caller).
        freq : str
            Frequency string (e.g. '15T') used for resampling.
        charging_data : pd.DataFrame
            The charging_data dataframe used to look up per-cdr metadata

        Returns
        -------
        dict
            Mapping of `cdr_id` to resampled and aligned DataFrame.
        """
        aligned_series = {}
        for key, df in time_series_dict.items():
            freq = f'{min_resolution}min'
            # Align start/end to the nearest quarter for stable resampling

            start = _round_to_nearest_quarter(df['datetime'].iloc[0])
            end=_round_to_nearest_quarter(df['datetime'].iloc[-1])
            end_idle = _round_to_nearest_quarter(pd.Timestamp(charging_data[charging_data.cdr_id==key].end_time_idle.values[0]))
            if end_idle.timestamp()-end.timestamp()<0:
                # in some cases cdr records are incorrect --> time and power are not sufficient to match the transfered energy. Assuming that the energy and max power are correct, the time must be false
                end_idle=end
            date_range = pd.date_range(start=start, end=end, freq=freq)
            df = df.set_index('datetime')
            df = df[['measure_value','transfered_energy','time_diff','power']]

            resampled_df = df.resample(freq, closed='right').mean()

            if resampled_df.power.isna().any():
                resampled_df.transfered_energy = resampled_df.transfered_energy.interpolate()
            
            if len(date_range)>len(resampled_df):
                new_row = pd.DataFrame({
                    # 'measure_date': [end],
                    'measure_value': [df['measure_value'].iloc[-1]],  # Use last known value or adjust as needed
                    'transfered_energy': [df['transfered_energy'].iloc[-1]],  # Use last known value or adjust as needed
                    'time_diff': [df['time_diff'].iloc[-1]],  # Set to zero or appropriate value
                    'power': [0.0]  # Set to zero or appropriate value
                })
                new_row['datetime'] = pd.to_datetime([end])
                new_row.set_index('datetime', inplace=True)
                # resampled_df = resampled_df.append(new_row) Deprecated
                resampled_df = pd.concat([resampled_df, new_row], ignore_index=True)
                resampled_df.sort_index(inplace=True)
                
            elif len(date_range)<len(resampled_df):
                if resampled_df.index[0]==date_range[0]:
                    resampled_df.drop(resampled_df.index[-1], inplace=True)
                elif resampled_df.index[-1]==date_range[-1]:
                    resampled_df.drop(resampled_df.index[0], inplace=True)
                # or other aggregation function
            # assert resampled_df.power.iloc[-1]!=0
            resampled_df = resampled_df.reset_index()
            resampled_df['datetime']=date_range
            resampled_df = resampled_df.set_index('datetime')
            resampled_df['occupancy']=(resampled_df.power>0).astype(int)
            resampled_df['occupancy_idle']=(resampled_df.power>0).astype(int)
            resampled_df= resampled_df.reindex(master_index, fill_value=0)
            resampled_df = resampled_df.fillna(0)
            resampled_df['occupancy_idle'] = ((resampled_df.index >= start) & (resampled_df.index <= end_idle)).astype(int)
            aligned_series[key] = resampled_df

            assert (resampled_df.loc[resampled_df['occupancy']==1, 'occupancy_idle']==1).all()
            assert resampled_df.occupancy.count() <= resampled_df.occupancy_idle.count()
            assert isinstance(resampled_df, pd.DataFrame)
        return aligned_series
    
    def sum_aligned_series(aligned_series):
        """Sum a mapping of aligned DataFrames element-wise.

        The function expects all DataFrames to share the same index and
        compatible columns so that the Python sum() operator produces the
        element-wise aggregation.
        """
        final_series = sum(aligned_series.values())
        return final_series

    # Step 1: Resample and align all time series
    aligned_series = resample_and_align(charging_data_meter_dict, time_range, min_resolution, charging_data)
    
    # Step 2: Sum the aligned time series
    result = sum_aligned_series(aligned_series)

    # Step 3: Get the results
    result_power = result.power.values
    result_occupancy = result.occupancy.values
    result_occupancy_idle = result.occupancy_idle.values

    # Do the same steps again to extract only dc profiles
    charging_dc = charging_data[charging_data['max_socket_power']>22]
    charging_dc = charging_dc[charging_dc['cdr_id'].isin(aligned_series.keys())]
    if charging_dc.shape[0]>0:
        aligned_series_dc = {}
        for i, row in charging_dc.iterrows():
            aligned_series_dc[row.cdr_id] = aligned_series.get(row.cdr_id)

        result_dc = sum_aligned_series(aligned_series_dc)
        result_power_dc = result_dc.power.values
        result_occupancy_dc = result_dc.occupancy.values
        result_occupancy_dc_idle = result_dc.occupancy_idle.values
    else:
        result_power_dc = np.zeros(len(result_time))
        result_occupancy_dc = np.zeros(len(result_time))
        result_occupancy_dc_idle = np.zeros(len(result_time))



    p_demand_timeseries = pd.DataFrame(list(zip(result_time, result_power, result_occupancy_idle, result_occupancy, result_power_dc, result_occupancy_dc_idle, result_occupancy_dc)),
                                       columns=['time', 'power_demand', 'cluster_occupancy_idle', 'cluster_occupancy_charging', 'power_demand_dc', 'cluster_occupancy_dc_idle','cluster_occupancy_dc_charging'])

    assert not (p_demand_timeseries.power_demand<0).any()
    assert (p_demand_timeseries.cluster_occupancy_idle>=p_demand_timeseries.cluster_occupancy_charging).all()
    assert (p_demand_timeseries[p_demand_timeseries['cluster_occupancy_idle']==0].power_demand==0).all()
    
    charging_data['charging_end'] = 0
    charging_data['charging_end'] = pd.to_datetime(charging_data['charging_end'], errors='coerce')

    for i, row in charging_data[charging_data['charging_end']==0].iterrows():
        charging_data.loc[i,'charging_end']=charging_data_meter_dict.get(row.cdr_id).datetime.iloc[-1]
    # Localize to Central European Time (CET/CEST)
    cet = pytz.timezone('Europe/Berlin')
    if not pd.api.types.is_datetime64tz_dtype(charging_data['end_time_idle']) and localize_dt:
        charging_data['end_time_idle'] = pd.to_datetime(charging_data['end_time_idle']).dt.tz_localize(cet)
    if localize_dt:
        charging_data['charging_end'] = pd.to_datetime(charging_data['charging_end']).dt.tz_localize(cet)
    
    if not (charging_data['end_time_idle']>=charging_data['charging_end']).all():
        # Set 'end_time_idle' to 'charging_end' where 'end_time_idle' < 'charging_end'
        charging_data.loc[charging_data['end_time_idle'] < charging_data['charging_end'], 'end_time_idle'] = charging_data['charging_end']

    assert (charging_data['end_time_idle']>=charging_data['charging_end']).all()
    charging_data['end_time']=charging_data['charging_end']
    
    return p_demand_timeseries, charging_data

def process_weekly_data(df, data_path, filename):
    """Process a raw cluster timeseries into weekly averaged CSV output.

    This function extracts `power_demand` and `cluster_occupancy_charging`
    from the provided DataFrame, resamples to hourly resolution, computes
    weekly aggregates, and writes the result to a CSV under
    `{data_path}/power_demands_analysis/weekly_average/`.
    """
    global DATANAME
    DATANAME = 'power_demand'
    data = df[['time','power_demand']].set_index('time').rename(columns={'power_demand': 'data'}).resample('1h').mean()
    # Extend data with zeros to ensure the shape of 168 is reached
    if data.shape[0] < 168:
        missing_rows = 168 - data.shape[0]
        last_index = data.index[-1]
        new_index = pd.date_range(start=last_index + pd.Timedelta(hours=1), periods=missing_rows, freq='h')
        zero_data = pd.DataFrame(0, index=new_index, columns=data.columns)
        data = pd.concat([data, zero_data])
    power_data_result = _get_weekly_data(data, DATANAME)

    DATANAME = 'occupancy' 
    data = df[['time','cluster_occupancy_charging']].set_index('time').rename(columns={'cluster_occupancy_charging': 'data'}).resample('1h').mean()
    if data.shape[0] < 168:
        missing_rows = 168 - data.shape[0]
        last_index = data.index[-1]
        new_index = pd.date_range(start=last_index + pd.Timedelta(hours=1), periods=missing_rows, freq='h')
        zero_data = pd.DataFrame(0, index=new_index, columns=data.columns)
        data = pd.concat([data, zero_data])
    occup_data_result = _get_weekly_data(data, DATANAME)

    total_data = pd.merge(power_data_result, occup_data_result, how='outer', left_index=True, right_index=True)
    total_data = _sort_df(total_data,'weekly_average').reset_index()
    os.makedirs(f'{data_path}/power_demands_analysis/weekly_average/', exist_ok=True)
    total_data.to_csv(f'{data_path}/power_demands_analysis/weekly_average/{filename}_weekly_average.csv', index=True)
    return total_data

def process_weekday_weekend_data(df, data_path, filename):
    """Process a raw cluster timeseries into weekday/weekend averaged CSV.

    Similar to `process_weekly_data` but produces a weekday/weekend
    aggregation and writes the CSV to
    `{data_path}/power_demands_analysis/weekday_weekend_average/`.
    """
    global DATANAME
    DATANAME = 'power_demand'
    data = df[['time','power_demand']].set_index('time').rename(columns={'power_demand': 'data'}).resample('1h').mean()
    if data.shape[0] < 168:
        missing_rows = 168 - data.shape[0]
        last_index = data.index[-1]
        new_index = pd.date_range(start=last_index + pd.Timedelta(hours=1), periods=missing_rows, freq='h')
        zero_data = pd.DataFrame(0, index=new_index, columns=data.columns)
        data = pd.concat([data, zero_data])
    power_data_result = get_weekday_weekend_data(data, DATANAME)

    DATANAME = 'occupancy' 
    data = df[['time','cluster_occupancy_charging']].set_index('time').rename(columns={'cluster_occupancy_charging': 'data'}).resample('1h').mean()
    if data.shape[0] < 168:
        missing_rows = 168 - data.shape[0]
        last_index = data.index[-1]
        new_index = pd.date_range(start=last_index + pd.Timedelta(hours=1), periods=missing_rows, freq='h')
        zero_data = pd.DataFrame(0, index=new_index, columns=data.columns)
        data = pd.concat([data, zero_data])
    occup_data_result = get_weekday_weekend_data(data, DATANAME)

    total_data = pd.merge(power_data_result, occup_data_result, how='outer', left_index=True, right_index=True)
    os.makedirs(f'{data_path}/power_demands_analysis/weekday_weekend_average/', exist_ok=True)
    total_data = _sort_df(total_data, 'weekday_weekend_average').reset_index()
    total_data.to_csv(f'{data_path}/power_demands_analysis/weekday_weekend_average/{filename}_weekday_weekend_average.csv', index=True)
    return total_data

def get_weekday_weekend_data(data, dataname=None):
    """Compute aggregated weekday/weekend statistics for a timeseries.

    Produces grouping that separates weekdays (as 'Weekday HH:MM') from
    weekend days and returns the mean of multiple summary statistics.
    """
    data_temp = _generate_weekday_weekend_data(data, dataname=dataname)
    return _get_weekday_weekend_group(data_temp).mean()

def _generate_weekday_weekend_data(data, dataname=None):
    """Generate aggregates for weekday/weekend grouping.

    Returns a combined DataFrame containing max/min/median/average and
    non-zero-day-median for each group produced by
    `_get_weekday_weekend_group`.
    """
    group = _get_weekday_weekend_group(data)
    data_max = _get_max_value(group, dataname=dataname)
    data_min = _get_min_value(group, dataname=dataname)
    data_med = _get_med_value(group, dataname=dataname)
    data_avg = _get_avg_value(group, dataname=dataname)
    data_non_zero_day_med = _get_non0_med_value(data, 'weekday_weekend', dataname=dataname)
    weekday_weekend_data = _merge_values(data_max, data_med, data_min, data_avg, data_non_zero_day_med)
    weekday_weekend_data = _sort_weekday_weekend_data(weekday_weekend_data)
    return weekday_weekend_data

def _get_weekday_weekend_group(data_pd):
    """Group a timeseries into 'Weekday HH:MM' and weekend-day groups.

    Weekdays are consolidated under the label 'Weekday HH:MM' while
    Saturday/Sunday remain separate (e.g. 'Sat 12:00').
    """
    def _separate_weekday_and_weekend(date):
        if (date.strftime('%a') in WEEKDAY):
            return date.strftime('Weekday %H:%M')
        else:
            return date.strftime('%a %H:%M')

    group = data_pd.groupby(_separate_weekday_and_weekend, sort=False)
    return group

def _sort_weekday_weekend_data(data):
    """Sort a weekday/weekend aggregated DataFrame into chronological order.

    The function converts group keys like 'Weekday HH:MM', 'Sat HH:MM',
    'Sun HH:MM' into concrete datetimes anchored on `MONDAY` to impose an
    ordering then sets the datetime index.
    """
    data['date'] = data.index
    data.reset_index(level=0, inplace=True, drop=True)

    def calc_date(data_row):
        day_of_week = data_row['date'].split(' ')[0]
        time_of_day = data_row['date'].split(' ')[1]
        hours = int(time_of_day.split(":")[0])
        minuets = int(time_of_day.split(":")[1])
        if 'Weekday' in day_of_week:
            return MONDAY + timedelta(days=4, hours=hours, minutes=minuets)
        else:
            return MONDAY + timedelta(days=int(WEEKLIST.index(day_of_week)), hours=hours,
                                            minutes=minuets)

    data['date'] = data.apply(calc_date, axis=1)
    data = data.sort_values(by='date')
    data.set_index('date', drop=True, inplace=True)
    return data

def _sort_df(df, timeframe):
    """Sort aggregated dataframes according to the requested timeframe.

    The function adds a temporary ordering column which is removed before
    returning. Supported timeframes: daily_average, holiday_average,
    weekend_average, weekday_weekend_average, weekly_average.
    """
    assert timeframe in ['daily_average', 'holiday_average', 'weekend_average','weekday_weekend_average', 'weekly_average'], 'Fleet Clustering: Currently only the following values are supported for the timeframe parameter: daily_average, holiday_average weekday_weekend_average, weekly_average'
    df = df.reset_index()
    if timeframe=='daily_average':
        df['day_order']=np.zeros(df.shape[0])
    elif timeframe == 'weekend_average':
        order_mapping = {'Sat': 1, 'Sun': 2}
        df['day_order'] = df['date'].str.extract(r'^(Sat|Sun)')[0].map(order_mapping)
        # Extract the hour and minute as a sortable time
        
    elif timeframe == 'weekday_weekend_average':
        order_mapping = {'Weekday': 0, 'Sat': 1, 'Sun': 2}
        df['day_order'] = df['date'].str.extract(r'^(Weekday|Sat|Sun)')[0].map(order_mapping)

    elif timeframe == 'weekly_average':
        order_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu':3, 'Fri':4,'Sat':5,'Sun':6}
        df['day_order'] = df['date'].str.extract(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)')[0].map(order_mapping)

    df['time'] = pd.to_datetime(df['date'].str.extract(r'(\d{2}:\d{2})')[0], format='%H:%M').dt.time
    # Sort by day order and then by time
    df = df.sort_values(by=['day_order', 'time']).drop(columns=['day_order', 'time'])
    return df

def _use_dc_profiles(row, base_path):
    """Load a DC standard power curve for the given `row` and scale it.

    Parameters
    ----------
    row : pandas.Series
        A record containing at least `max_socket_power` and `quantity_in_wh`.
    base_path : str
        Base repository path used to locate stored power curves.

    Returns
    -------
    (pd.DataFrame, datetime.timedelta)
        Tuple of the scaled DC charging profile (with columns `time`,
        `power`, `energy` etc.) and the resulting charge duration.
    """

    power_rating = row.max_socket_power
    charged_energy = row.quantity_in_wh / 1000.0

    # Map socket power ranges to filenames. The mapping should cover the
    # expected power rating ranges used in the dataset.
    mapping = {
        (22, 100): 'power_curves_100kW.csv',
        (100, 200): 'power_curves_200kW.csv',
        (200, 1000): 'power_curves__kW.csv',
    }

    filename = None
    for range_, mapped_value in mapping.items():
        if range_[0] <= power_rating < range_[1]:
            filename = mapped_value
            break

    curves_dir = f'{base_path}/data/DC_standard_power_curves'
    file_path = os.path.join(curves_dir, filename)
    dc_charging_profile = pd.read_csv(file_path)

    # Scale the reference power curve to the actual socket rating
    dc_charging_profile['power'] = (
        dc_charging_profile.power * power_rating / dc_charging_profile.power.max()
    )

    # Ensure 'time' is a datetime and compute cumulative energy (Wh -> kWh)
    dc_charging_profile['time'] = pd.to_datetime(dc_charging_profile['time'])
    time_diff = (dc_charging_profile['time'].shift(-1) - dc_charging_profile['time']).fillna(pd.Timedelta(seconds=0))
    dc_charging_profile['energy'] = (dc_charging_profile['power'] * time_diff.dt.total_seconds()).cumsum()
    dc_charging_profile['energy'] = (dc_charging_profile['energy'] - dc_charging_profile['energy'].iloc[0]) / 1000.0

    # Interpolate to 1-second resolution and cut at requested energy
    dc_charging_profile = dc_charging_profile.set_index('time').resample('1s').interpolate()
    dc_charging_profile = dc_charging_profile[dc_charging_profile['energy'] < charged_energy]
    dc_charging_profile = dc_charging_profile.reset_index(drop=False)

    charge_time = dc_charging_profile['time'].iloc[-1] - dc_charging_profile['time'].iloc[0]

    return dc_charging_profile, charge_time

def _get_total_seconds(td):
    """Return the total seconds contained in a timedelta-like object.

    Wrapper around ``timedelta.total_seconds()`` to make intent explicit.
    """
    return td.total_seconds()

def _strip_timezone_column(series):
    return series.map(_strip_timezone)

def _strip_timezone(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp

def _round_to_nearest_quarter(dt):
    """Round a datetime to the nearest 15-minute interval.

    Rounds to the closest quarter hour. If exactly half-way, rounds up.
    """
    # Get the number of minutes past the hour
    minutes = dt.minute
    # Calculate the remainder when divided by 15 (a quarter of an hour)
    remainder = minutes % 15

    # Calculate the number of minutes to add or subtract
    if remainder < 7.5:
        delta_minutes = -remainder
    else:
        delta_minutes = 15 - remainder

    # Create a timedelta object for the number of minutes to add/subtract
    delta = datetime.timedelta(minutes=delta_minutes)
    
    # Round the datetime
    rounded_dt = dt + delta
    
    # Set seconds and microseconds to zero for precise quarter hour
    rounded_dt = rounded_dt.replace(second=0, microsecond=0)
    
    return rounded_dt

def _get_weekly_data(data, dataname=None):
    """Compute aggregated weekly statistics for a timeseries.

    The function groups the input series by weekday and time-of-day then
    returns the mean of the grouped aggregates (max, min, median, average,
    and median over non-zero days) computed in `_generate_weekly_data`.
    """
    data_temp = _generate_weekly_data(data, dataname=dataname)
    return _get_weekly_group(data_temp).mean()

def _generate_weekly_data(data, dataname=None):
    """Generate multiple weekly aggregates (max/min/median/avg/non-zero-med).

    Returns a combined DataFrame containing several summary columns for
    each weekday/time slot. The result is then sorted by weekday and time.
    """
    group = _get_weekly_group(data)
    data_max = _get_max_value(group, dataname=dataname)
    data_min = _get_min_value(group, dataname=dataname)
    data_med = _get_med_value(group, dataname=dataname)
    data_avg = _get_avg_value(group, dataname=dataname)
    data_non_zero_day_med = _get_non0_med_value(data, 'weekly', dataname=dataname)
    weekly_data = _merge_values(data_max, data_med, data_min, data_avg, data_non_zero_day_med)
    weekly_data = _sort_weekly_data(weekly_data)
    return weekly_data

def _get_weekly_group(data_pd):
    """Group a daily timeseries by weekday and clock-time.

    The grouping key is a string like 'Mon 12:00' produced with
    ``strftime('%a %H:%M')``. This helper is used for weekly aggregations.
    """
    group = data_pd.groupby(lambda x: x.strftime('%a %H:%M'), sort=False)
    return group

def _get_max_value(group, dataname=None):
        """Get the max values of the grouped data
        """
        data_max = group.max()
        data_max.rename(
            columns={'data': f'Max_{dataname}'}, inplace=True)
        return data_max

def _get_min_value(group, dataname=None):
    """Get the min values of the grouped data
    """
    data_min = group.min()
    data_min.rename(
        columns={'data': f'Min_{dataname}'}, inplace=True)
    return data_min

def _get_med_value(group, dataname=None):
    """Get the median values of the grouped data
    """
    data_med = group.median()
    data_med.rename(
        columns={'data': f'Median_{dataname}'}, inplace=True)
    return data_med

def _get_avg_value(group, dataname=None):
    """Get the average values of the grouped data
    """
    data_med = group.mean()
    data_med.rename(
        columns={'data': f'Average_{dataname}'}, inplace=True)
    return data_med

def _get_non0_med_value(data, timeframe, dataname=None):
    """Get the median values of the grouped data from days, where the total sum of transfered energy is greater than zero
    """
    df_help = data.resample('D').sum()
    df_help = df_help[df_help["data"]==0]
    indices_to_remove = df_help.index
    data = data[~data.index.isin(indices_to_remove)]

    if timeframe == 'weekly':
        group = _get_weekly_group(data)
    elif timeframe == 'weekday_weekend':
        group = _get_weekday_weekend_group(data)
    
    data_med = group.median()
    data_med.rename(
        columns={'data': f'Median_non_zero_day_{dataname}'}, inplace=True)
    return data_med

def _merge_values( *data):
    """merge min. max. and mean. data into one dataframe

    Args:
        *data :  dataframes

    Returns:
        dataframe: data after merging
    """
    data_result = pd.concat(data, axis=1)
    return data_result

def _sort_weekly_data( data):
    """sort data according to date, date format is "Mon 12:00"
    """
    monday = MONDAY
    weekday_list = WEEKLIST

    # delet index, instead add column date
    data['date'] = data.index
    data.reset_index(level=0, inplace=True, drop=True)

    def calc_date(data_row):
        day_of_week = data_row['date'].split(' ')[0]
        time_of_day = data_row['date'].split(' ')[1]
        hours = int(time_of_day.split(":")[0])
        minuets = int(time_of_day.split(":")[1])
        return monday + timedelta(days=int(weekday_list.index(day_of_week)), hours=hours, minutes=minuets)

    data['date'] = data.apply(calc_date, axis=1)
    data = data.sort_values(by='date')
    data.set_index('date', drop=True, inplace=True)
    return data




