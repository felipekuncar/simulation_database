# Writen by Felipe Kuncar
# felipe.kuncar@canterbury.ac.nz
# Last modified on 03-11-2025

#======================================================================================================================
# IMPORT
#======================================================================================================================

from pathlib import Path
import csv
import pandas as pd
import xarray as xr
import numpy as np

#======================================================================================================================
# SELECT COMPUTER
#======================================================================================================================

computer = 'local'
#computer = 'RCH'

#======================================================================================================================
# SELECT OPTION
#======================================================================================================================

# Only finite-fault models
model_option = 'ff'
# Both finite-fault and point-source models
#model_option = 'ff_ps'

#======================================================================================================================
# PATHS
#======================================================================================================================

if computer == 'local':

    nzgmdb_rotd50_path = Path(__file__).parents[6] / '21_NZGMDB' / 'NZGMDB_v4p3' / 'NZGMDB-Quality-Flatfiles' / 'Intensity-Measure Ground-Motion Flat-Files' / 'ground_motion_im_table_rotd50_flat.csv'
    nzgmdb_geom_path = Path(__file__).parents[6] / '21_NZGMDB' / 'NZGMDB_v4p3' / 'NZGMDB-Quality-Flatfiles' / 'Intensity-Measure Ground-Motion Flat-Files' / 'ground_motion_im_table_geom_flat.csv'
    nzgmdb_eas_path = Path(__file__).parents[6] / '21_NZGMDB' / 'NZGMDB_v4p3' / 'NZGMDB-Quality-Flatfiles' / 'Intensity-Measure Ground-Motion Flat-Files' / 'ground_motion_im_table_eas_flat.csv'
    sim_path = Path(__file__).parents[1] / 'results'

elif computer == 'RCH':

    nzgmdb_rotd50_path = Path(__file__).parents[1] / 'NZGMDB' / 'NZGMDB_v4p3' / 'NZGMDB-Quality-Flatfiles' / 'Intensity-Measure Ground-Motion Flat-Files' / 'ground_motion_im_table_rotd50_flat.csv'
    nzgmdb_geom_path = Path(__file__).parents[1] / 'NZGMDB' / 'NZGMDB_v4p3' / 'NZGMDB-Quality-Flatfiles' / 'Intensity-Measure Ground-Motion Flat-Files' / 'ground_motion_im_table_geom_flat.csv'
    nzgmdb_eas_path = Path(__file__).parents[1] / 'NZGMDB' / 'NZGMDB_v4p3' / 'NZGMDB-Quality-Flatfiles' / 'Intensity-Measure Ground-Motion Flat-Files' / 'ground_motion_eas_table_geom_flat.csv'
    sim_path = Path('/scratch/projects/rch-quakecore/Felipe_Validation')

#======================================================================================================================
# READ OBSERVATIONAL DATABASE
#======================================================================================================================

nzgmdb_rotd50_df = pd.read_csv(nzgmdb_rotd50_path, dtype={'evid': str, 'sta': str})
nzgmdb_geom_df = pd.read_csv(nzgmdb_geom_path, dtype={'evid': str, 'sta': str})
nzgmdb_eas_df = pd.read_csv(nzgmdb_eas_path, dtype={'evid': str, 'sta': str})

#======================================================================================================================
# SELECT EVENTS
#======================================================================================================================

'''
# Option 1:
# Provide a list of events
event_list = ['2016p858508', '2019p315560', '2024p945089', '2015p127696']
'''

# Option 2
if model_option == 'ff_ps':
    # Obtain a list of events for which simulations are available for both finite-fault and point-source models
    # Get all folder names
    folder_names = [f.name for f in sim_path.iterdir() if f.is_dir()]
    # Separate those with and without "_p" (finite-fault and point-source models, respectively)
    base_names = {name for name in folder_names if not name.endswith("_p")}
    p_names = {name[:-2] for name in folder_names if name.endswith("_p")}
    # Find the intersection — names that have both versions
    event_list = sorted(base_names & p_names)
elif model_option == 'ff':
    # Get all folder names
    folder_names = [f.name for f in sim_path.iterdir() if f.is_dir()]
    # select those without "_p" (finite-fault models)
    event_list = {name for name in folder_names if not name.endswith("_p")}

print('Event list:', event_list)

#======================================================================================================================
# EXTRACT AVAILABLE SIMULATIONS AT STATIONS WITH RECORDINGS
#======================================================================================================================

sim_data = []

for evid in event_list:

    print('Extracting data from event', evid)

    # Observational data available for this event
    nzgmdb_rotd50_event_df = nzgmdb_rotd50_df[nzgmdb_rotd50_df['evid'] == evid]
    nzgmdb_geom_event_df = nzgmdb_geom_df[nzgmdb_geom_df['evid'] == evid]

    if model_option == 'ff_ps':
        # Point-source model id
        evid_p = f'{evid}_p'

    if computer == 'local':
        # Finite-fault model
        sim_ims = xr.open_dataset(sim_path / evid / 'sim_output' / 'intensity_measures.h5')
        if model_option == 'ff_ps':
            # Point-source model
            sim_ims_p = xr.open_dataset(sim_path / evid_p / 'sim_output' / 'intensity_measures.h5')

    elif computer == 'RCH':
        ### RCH
        # Finite-fault model
        sim_ims = xr.open_dataset(sim_path / evid / 'intensity_measures.h5')
        if model_option == 'ff_ps':
            # Point-source model
            sim_ims_p = xr.open_dataset(sim_path / evid_p / 'intensity_measures.h5')

    # Stations simulated in each model
    ff_stations = pd.Index(sim_ims.station.values)
    # Stations with simulations and recordings
    stations = ff_stations.intersection(nzgmdb_rotd50_event_df['sta'])
    if model_option == 'ff_ps':
        ps_stations = pd.Index(sim_ims_p.station.values)
        # Intersection: only stations simulated in both models
        common_sim_stations = ff_stations.intersection(ps_stations)
        # Stations with simulations (both models) and recordings
        stations = common_sim_stations.intersection(nzgmdb_rotd50_event_df['sta'])

    print(f'Number of stations with simulations (both models) and recordings: {len(stations)}')

    if len(stations) == 0:
        print(f'No common stations for event {evid}. Skipping.')
        continue

    for sta in stations:

        sim_data.append({'evid': evid, 'sta': sta})

# Data frame with events and stations with both simulations and recordings
sim_data_df = pd.DataFrame(sim_data)

# =====================================================================================================================
# REMOVE SITES AND EVENT THAT DO NOT MEET MINIMUM REQUIREMENTS
# =====================================================================================================================

# Define the minimum requirements
min_GMs_perStation = 3
min_GMs_perEvent = 3

# Function to remove stations and events according to the requirements
def filter_stations_and_events(df):
    while True:
        # Remove stations with fewer than 3 events
        station_counts = df['sta'].value_counts()
        stations_to_keep = station_counts[station_counts >= min_GMs_perStation].index
        df = df[df['sta'].isin(stations_to_keep)]
        # Remove events recorded in fewer than 3 stations
        event_counts = df['evid'].value_counts()
        events_to_keep = event_counts[event_counts >= min_GMs_perEvent].index
        df = df[df['evid'].isin(events_to_keep)]
        # Check if further filtering is needed
        new_station_counts = df['sta'].value_counts()
        new_event_counts = df['evid'].value_counts()
        if (station_counts.equals(new_station_counts)) and (event_counts.equals(new_event_counts)):
            break
    return df

# Filter the dataframe
sim_data_df = filter_stations_and_events(sim_data_df)

print('Number of sites considered:', len(sim_data_df['sta'].drop_duplicates()))
print('Number of events considered:', len(sim_data_df['evid'].drop_duplicates()))
print('Number of ground motions considered:', len(sim_data_df['sta']))

#======================================================================================================================
# EXTRACT VIBRATION PERIODS FOR WHICH RESPONSE SPECTRA IS AVAILABLE
#======================================================================================================================

# Read one simulated event and extract the vibration periods
first_evid = sim_data_df.iloc[0]['evid']
T_array_sim = sim_ims.period.values

# Extract vibration periods in observational database
pSA_cols = [col for col in nzgmdb_rotd50_df.columns if col.startswith('pSA')]
T_array_obs = np.array([float(col.split('_')[1]) for col in pSA_cols])

# Use the vibration periods from simulations
T_array = T_array_sim

#======================================================================================================================
# EXTRACT FREQUENCIES FOR WHICH FAS IS AVAILABLE
#======================================================================================================================

# Read one simulated event and extract the vibration periods
first_evid = sim_data_df.iloc[0]['evid']
f_array_sim = sim_ims.frequency.values

# Extract vibration periods in observational database
f_cols = [col for col in nzgmdb_eas_df.columns if col.startswith('FAS')]
f_array_obs = np.array([float(col.split('_')[1]) for col in f_cols])

# Use the vibration periods from simulations
f_array = f_array_sim

#======================================================================================================================
# CREATE CSV FILES FOR RESIDUAL ANALYSIS AND WRITE HEADERS
#======================================================================================================================

# Create folder to save files
if model_option == 'ff':
    residual_input_path = Path(__file__).parent / 'residual_input_ff'
    residual_input_path.mkdir(exist_ok=True)
elif model_option == 'ff_ps':
    residual_input_path = Path(__file__).parent / 'residual_input_ff_ps'
    residual_input_path.mkdir(exist_ok=True)

# Stations
f_stations = open(residual_input_path / 'stations.csv', 'w', newline="")
writer_stations = csv.writer(f_stations)
writer_stations.writerow(['stat_id', 'stat_name'])

# Events
f_events = open(residual_input_path / 'events.csv', 'w', newline="")
writer_events = csv.writer(f_events)
writer_events.writerow(['event_id', 'event_name'])

# Simulated and observed intensity measures
# Define header
im_header = ['gm_id', 'event_id', 'stat_id', 'PGA', 'PGV', 'CAV', 'AI', 'Ds575', 'Ds595']
for T in T_array:
    im_header.append('pSA_%.12f' % T)
for f in f_array:
    im_header.append('EAS_%.12f' % f)
# Create CSV file and write the header
f_im_sim = open(Path(residual_input_path / 'im_sim.csv'), 'w', newline="")
writer_im_sim = csv.writer(f_im_sim)
writer_im_sim.writerow(im_header)
if model_option == 'ff_ps':
    f_im_sim_p = open(Path(residual_input_path / 'im_sim_p.csv'), 'w', newline="")
    writer_im_sim_p = csv.writer(f_im_sim_p)
    writer_im_sim_p.writerow(im_header)
f_im_obs = open(Path(residual_input_path / 'im_obs.csv'), 'w', newline="")
writer_im_obs = csv.writer(f_im_obs)
writer_im_obs.writerow(im_header)

#======================================================================================================================
# FILL THE FILES WITH DATA AND SAVE THEM
#======================================================================================================================

# Initialize counters
event_id = 1
stat_id = 1
gm_id = 1
event_dict = {}
station_dict = {}

for idx, row in sim_data_df.iterrows():

    event_name = row['evid']
    stat_name = row['sta']

    # Assign unique event_id
    if event_name not in event_dict:
        event_dict[event_name] = event_id
        writer_events.writerow([event_id, event_name])
        event_id += 1
    current_event_id = event_dict[event_name]

    # Assign unique stat_id
    if stat_name not in station_dict:
        station_dict[stat_name] = stat_id
        writer_stations.writerow([stat_id, stat_name])
        stat_id += 1
    current_stat_id = station_dict[stat_name]

    ### Fill simulated IMs file
    # Read simulation results
    if computer == 'local':
        # Finite-fault model
        sim_ims = xr.open_dataset(sim_path / event_name / 'sim_output' / 'intensity_measures.h5')
        if model_option == 'ff_ps':
            # Point-source model
            sim_ims_p = xr.open_dataset(sim_path / f'{event_name}_p' / 'sim_output' / 'intensity_measures.h5')
    elif computer == 'RCH':
        ### RCH
        # Finite-fault model
        sim_ims = xr.open_dataset(sim_path / event_name / 'intensity_measures.h5')
        if model_option == 'ff_ps':
            # Point-source model
            sim_ims_p = xr.open_dataset(sim_path / f'{event_name}_p' / 'intensity_measures.h5')
    # Extract values
    PGA = sim_ims.PGA.sel(station=stat_name, component='rotd50').values
    PGV = sim_ims.PGV.sel(station=stat_name, component='rotd50').values
    CAV = sim_ims.CAV.sel(station=stat_name, component='geom').values
    AI = sim_ims.AI.sel(station=stat_name, component='geom').values
    Ds575 = sim_ims.Ds575.sel(station=stat_name, component='geom').values
    Ds595 = sim_ims.Ds595.sel(station=stat_name, component='geom').values
    SA_array = sim_ims.pSA.sel(station=stat_name, component='rotd50', period=T_array).to_dataframe()['pSA'].values
    FAS_000_array = sim_ims.FAS.sel(station=stat_name, component='000', frequency=f_array).to_dataframe()['FAS'].values
    FAS_090_array = sim_ims.FAS.sel(station=stat_name, component='090', frequency=f_array).to_dataframe()['FAS'].values
    EAS_array = np.sqrt(0.5 * (FAS_000_array ** 2 + FAS_090_array ** 2))
    # Save values
    im_sim_values = [gm_id, current_event_id, current_stat_id, PGA, PGV, CAV, AI, Ds575, Ds595] + SA_array.tolist() + EAS_array.tolist()

    if model_option == 'ff_ps':
        ### Fill point-source simulated IMs file
        # Read simulation results
        if computer == 'local':
            # Point-source model
            sim_ims_p = xr.open_dataset(sim_path / f'{event_name}_p' / 'sim_output' / 'intensity_measures.h5')
        elif computer == 'RCH':
            # Point-source model
            sim_ims_p = xr.open_dataset(sim_path / f'{event_name}_p' / 'intensity_measures.h5')
        # Extract values
        PGA = sim_ims_p.PGA.sel(station=stat_name, component='rotd50').values
        PGV = sim_ims_p.PGV.sel(station=stat_name, component='rotd50').values
        CAV = sim_ims_p.CAV.sel(station=stat_name, component='geom').values
        AI = sim_ims_p.AI.sel(station=stat_name, component='geom').values
        Ds575 = sim_ims_p.Ds575.sel(station=stat_name, component='geom').values
        Ds595 = sim_ims_p.Ds595.sel(station=stat_name, component='geom').values
        SA_array = sim_ims_p.pSA.sel(station=stat_name, component='rotd50', period=T_array).to_dataframe()['pSA'].values
        FAS_000_array = sim_ims_p.FAS.sel(station=stat_name, component='000', frequency=f_array).to_dataframe()['FAS'].values
        FAS_090_array = sim_ims_p.FAS.sel(station=stat_name, component='090', frequency=f_array).to_dataframe()['FAS'].values
        EAS_array = np.sqrt(0.5 * (FAS_000_array ** 2 + FAS_090_array ** 2))
        # Save values
        im_sim_p_values = [gm_id, current_event_id, current_stat_id, PGA, PGV, CAV, AI, Ds575, Ds595] + SA_array.tolist() + EAS_array.tolist()

    ### Fill observed IMs file
    # Read observational datasets
    nzgmdb_rotd50_event_station_df = nzgmdb_rotd50_df[(nzgmdb_rotd50_df['evid'] == event_name) & (nzgmdb_rotd50_df['sta'] == stat_name)]
    nzgmdb_geom_event_station_df = nzgmdb_geom_df[(nzgmdb_geom_df['evid'] == event_name) & (nzgmdb_geom_df['sta'] == stat_name)]
    nzgmdb_eas_event_station_df = nzgmdb_eas_df[(nzgmdb_eas_df['evid'] == event_name) & (nzgmdb_eas_df['sta'] == stat_name)]
    # Extract values
    PGA = nzgmdb_rotd50_event_station_df['PGA'].values[0]
    PGV = nzgmdb_rotd50_event_station_df['PGV'].values[0]
    CAV = nzgmdb_geom_event_station_df['CAV'].values[0]
    AI = nzgmdb_geom_event_station_df['AI'].values[0]
    Ds575 = nzgmdb_geom_event_station_df['Ds575'].values[0]
    Ds595 = nzgmdb_geom_event_station_df['Ds595'].values[0]
    pSA_cols = [col for col in nzgmdb_rotd50_event_station_df.columns if col.startswith('pSA')]
    SA_array  = nzgmdb_rotd50_event_station_df[pSA_cols].iloc[0].values
    EAS_cols = [col for col in nzgmdb_eas_event_station_df.columns if col.startswith('FAS')]
    EAS_array  = nzgmdb_eas_event_station_df[EAS_cols].iloc[0].values
    # Interpolate SA and EAS values to be consistent with simulation array
    SA_array = np.interp(T_array, T_array_obs, SA_array)
    EAS_array = np.interp(f_array, f_array_obs, EAS_array)
    # Save values
    im_obs_values = [gm_id, current_event_id, current_stat_id, PGA, PGV, CAV, AI, Ds575, Ds595] + SA_array.tolist() + EAS_array.tolist()

    # Write to sim and obs CSVs
    writer_im_sim.writerow(im_sim_values)
    if model_option == 'ff_ps':
        writer_im_sim_p.writerow(im_sim_p_values)
    writer_im_obs.writerow(im_obs_values)

    gm_id += 1  # increment ground motion ID

# Close all files
f_stations.close()
f_events.close()
f_im_sim.close()
if model_option == 'ff_ps':
    f_im_sim_p.close()
f_im_obs.close()

print("CSV files created successfully in:", residual_input_path)