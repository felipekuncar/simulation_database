# Writen by Felipe Kuncar
# Last modified on 16-10-2025

from pathlib import Path
import xarray as xr
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
#import oq_wrapper as oqw

rootDir = Path(__file__).parents[6]

#nzgmdb_path = rootDir / '21_NZGMDB' / 'NZGMDB_v4p3' / 'quality_flatfiles_4p3' / 'quality_db' / 'ground_motion_im_table_rotd50_flat.csv'
nzgmdb_path = rootDir / '21_NZGMDB' / 'NZGMDB_v4p3' / 'quality_flatfiles_4p3' / 'quality_db' / 'ground_motion_im_table_geom_flat.csv'
nzgmdb_df = pd.read_csv(nzgmdb_path, dtype={'evid': str, 'sta': str})

event = '2013p544960'

obs_df = nzgmdb_df[nzgmdb_df['evid'] == event]

sim_ims = xr.open_dataset('sim_output/intensity_measures.h5')

station_ids = sim_ims.station.values


Kuncar2026_path = Path(__file__).parents[2] / 'Kuncar2026_sims' / 'v2' / 'Method1' / 'CB14a'/ 'Input'
events_K2026 = pd.read_csv(Kuncar2026_path / 'events.csv')
stations_K2026 = pd.read_csv(Kuncar2026_path / 'stations.csv')
im_obs_K2026 = pd.read_csv(Kuncar2026_path / 'im_obs.csv')
im_sim_K2026 = pd.read_csv(Kuncar2026_path / 'im_sim.csv')
cols_pSA = [col for col in im_sim_K2026.columns if col.startswith('pSA')]
periods_K2026 = np.array([float(col.split('_')[1]) for col in cols_pSA])
event_id = events_K2026[events_K2026['event_name'] == event]['event_id'].values[0]
print(event_id)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.15, hspace=0.15)

fig.suptitle(event, fontsize=18, y=0.98)

ax = fig.add_subplot(gs[0, 0])
#sim_df = sim_ims.pSA.sel(station=station_ids, component='rotd50', period=0.01).to_dataframe()
sim_df = sim_ims.pSA.sel(station=station_ids, component='geom', period=0.01).to_dataframe()
ax.scatter(sim_df['rrup'], sim_df['pSA'], s=2.0, color='k', label='Simulation')
ax.scatter(obs_df['r_rup'], obs_df['pSA_0.01'], s=3.0, color='r', label='Observation')
ax.legend(loc=3, fontsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.25, color='k')
ax.set_ylabel('SA(0.01s) (g)', size=16)
ax.set_xlim(10, 300)
ax.set_ylim(0.00001, 1)
ax.set_xscale('log')
ax.set_yscale('log')

ax = fig.add_subplot(gs[0, 1])
sim_df = sim_ims.pSA.sel(station=station_ids, component='rotd50', period=0.1).to_dataframe()
ax.scatter(sim_df['rrup'], sim_df['pSA'], s=2.0, color='k', label='Simulation')
ax.scatter(obs_df['r_rup'], obs_df['pSA_0.1'], s=3.0, color='r', label='Observation')
ax.set_ylabel('SA(0.1s) (g)', size=16)
ax.grid(True, which='both', linestyle='--', linewidth=0.25, color='k')
ax.set_xlim(10, 300)
ax.set_ylim(0.00001, 1)
ax.set_xscale('log')
ax.set_yscale('log')

ax = fig.add_subplot(gs[1, 0])
sim_df = sim_ims.pSA.sel(station=station_ids, component='rotd50', period=1).to_dataframe()
ax.scatter(sim_df['rrup'], sim_df['pSA'], s=2.0, color='k', label='Simulation')
ax.scatter(obs_df['r_rup'], obs_df['pSA_1.0'], s=3.0, color='r', label='Observation')
ax.set_xlabel('$R_{rup}$ (km)', size=16)
ax.set_ylabel('SA(1.0s) (g)', size=16)
ax.grid(True, which='both', linestyle='--', linewidth=0.25, color='k')
ax.set_xlim(10, 300)
ax.set_ylim(0.00001, 1)
ax.set_xscale('log')
ax.set_yscale('log')

ax = fig.add_subplot(gs[1, 1])
sim_df = sim_ims.pSA.sel(station=station_ids, component='rotd50', period=5).to_dataframe()
ax.scatter(sim_df['rrup'], sim_df['pSA'], s=2.0, color='k', label='Simulation')
ax.scatter(obs_df['r_rup'], obs_df['pSA_5.0'], s=3.0, color='r', label='Observation')
ax.set_xlabel('$R_{rup}$ (km)', size=16)
ax.set_ylabel('SA(5.0s) (g)', size=16)
ax.grid(True, which='both', linestyle='--', linewidth=0.25, color='k')
ax.set_xlim(10, 300)
ax.set_ylim(0.00001, 1)
ax.set_xscale('log')
ax.set_yscale('log')

fig.subplots_adjust(top=0.94, bottom=0.05, left=0.07, right=0.99)

#plt.show()
plt.savefig('plot1.png', dpi=600)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

'''
station_1 = 'MGCS'
station_2 = 'BWRS'
station_3 = 'QCCS'
station_4 = 'FKPS'
station_5 = 'PTOS'
station_6 = 'PVCS'
'''

station_1 = 'MGCS'
station_2 = 'QCCS'
station_3 = 'TFSS'
station_4 = 'WEMS'
station_5 = 'WNHS'
station_6 = 'LRSS'

cols_pSA = [col for col in obs_df.columns if col.startswith('pSA')]
periods = np.array([float(col.split('_')[1]) for col in cols_pSA])
print(periods)
#print(obs_df.loc[obs_df['sta'] == station_1, obs_df.filter(like='pSA').columns].to_numpy())

#print(obs_df[obs_df['sta' == station_1]])
fig = plt.figure(figsize=(22, 14))
gs = fig.add_gridspec(nrows=2, ncols=3, wspace=0.15, hspace=0.15)

fig.suptitle(event, fontsize=18, y=0.98)

ax = fig.add_subplot(gs[0, 0])
ax.text(0.012, 0.50, station_1, fontsize=14)
ax.text(0.012, 0.25, '$V_{S30}=%.0f~(m/s)$' % obs_df[obs_df['sta'] == station_1]['Vs30'].values[0], fontsize=14)
ax.text(0.012, 0.12, '$R_{rup}=%.0f~(km)$' % obs_df[(obs_df['sta'] == station_1) & (obs_df['evid'] == event)]['r_rup'].values[0], fontsize=14)
# Observation
ax.plot(periods, obs_df.loc[obs_df['sta'] == station_1, obs_df.filter(like='pSA').columns].to_numpy().flatten(), color='k', linewidth=2.5, label='Observation')
# Simulation NZVM v2.09
sim_psa = sim_ims.pSA.sel(station=[station_1], component='rotd50')
ax.plot(sim_psa.period.values, sim_psa.values[0], color='r', linewidth=2.5, label='Simulation (NZVM 2.09)')
# Simulation NZVM v2.02
station_id = stations_K2026[stations_K2026['stat_name'] == station_1]['stat_id'].values[0]
ax.plot(periods_K2026, im_sim_K2026[(im_sim_K2026['stat_id'] == station_id) & (im_sim_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='g', linewidth=2.5, label='Lee2022 simulation (NZVM 2.02)')
# Observation Lee et al. v2.02
ax.plot(periods_K2026, im_obs_K2026[(im_obs_K2026['stat_id'] == station_id) & (im_obs_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='gray', linewidth=2.5, linestyle=(2, (2, 2)), label='Lee2022 observation')
ax.legend(loc=3, fontsize=14)
#ax.grid(True, which='both', linestyle='--', linewidth=0.25, color='k')
ax.set_xlim(0.01, 10)
ax.set_ylim(0.00001, 1)
ax.set_xlabel('Vibration Period, T (s)', size=16)
ax.set_ylabel('Response Spectra, SA (g)', size=16)
ax.set_xscale('log')
ax.set_yscale('log')

ax = fig.add_subplot(gs[0, 1])
ax.text(0.012, 0.5, station_2, fontsize=14)
ax.text(0.012, 0.25, '$V_{S30}=%.0f~(m/s)$' % obs_df[obs_df['sta'] == station_2]['Vs30'].values[0], fontsize=14)
ax.text(0.012, 0.12, '$R_{rup}=%.0f~(km)$' % obs_df[(obs_df['sta'] == station_2) & (obs_df['evid'] == event)]['r_rup'].values[0], fontsize=14)
# Observation
ax.plot(periods, obs_df.loc[obs_df['sta'] == station_2, obs_df.filter(like='pSA').columns].to_numpy().flatten(), color='k', linewidth=2.5, label='Observation')
# Simulation NZVM v2.09
#sim_psa = sim_ims.pSA.sel(station=[station_2], component='rotd50')
sim_psa = sim_ims.pSA.sel(station=[station_2], component='geom')
ax.plot(sim_psa.period.values, sim_psa.values[0], color='r', linewidth=2.5, label='Simulation')
# Simulation NZVM v2.02
station_id = stations_K2026[stations_K2026['stat_name'] == station_2]['stat_id'].values[0]
ax.plot(periods_K2026, im_sim_K2026[(im_sim_K2026['stat_id'] == station_id) & (im_sim_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='g', linewidth=2.5, label='Lee2022 simulation (NZVM 2.02)')
# Observation Lee2022
ax.plot(periods_K2026, im_obs_K2026[(im_obs_K2026['stat_id'] == station_id) & (im_obs_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='gray', linewidth=2.5, linestyle=(2, (2, 2)), label='Lee2022 observation')
ax.set_xlim(0.01, 10)
ax.set_ylim(0.00001, 1)
ax.set_xlabel('Vibration Period, T (s)', size=16)
ax.set_ylabel('Response Spectra, SA (g)', size=16)
ax.set_xscale('log')
ax.set_yscale('log')

ax = fig.add_subplot(gs[0, 2])
ax.text(0.012, 0.5, station_3, fontsize=14)
ax.text(0.012, 0.25, '$V_{S30}=%.0f~(m/s)$' % obs_df[obs_df['sta'] == station_3]['Vs30'].values[0], fontsize=14)
ax.text(0.012, 0.12, '$R_{rup}=%.0f~(km)$' % obs_df[(obs_df['sta'] == station_3) & (obs_df['evid'] == event)]['r_rup'].values[0], fontsize=14)
# Observation
ax.plot(periods, obs_df.loc[obs_df['sta'] == station_3, obs_df.filter(like='pSA').columns].to_numpy().flatten(), color='k', linewidth=2.5, label='Observation')
# Simulation NZVM v2.09
#sim_psa = sim_ims.pSA.sel(station=[station_3], component='rotd50')
sim_psa = sim_ims.pSA.sel(station=[station_3], component='geom')
ax.plot(sim_psa.period.values, sim_psa.values[0], color='r', linewidth=2.5, label='Simulation')
# Simulation NZVM v2.02
station_id = stations_K2026[stations_K2026['stat_name'] == station_3]['stat_id'].values[0]
ax.plot(periods_K2026, im_sim_K2026[(im_sim_K2026['stat_id'] == station_id) & (im_sim_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='g', linewidth=2.5, label='Lee2022 simulation (NZVM 2.02)')
# Observation Lee2022
ax.plot(periods_K2026, im_obs_K2026[(im_obs_K2026['stat_id'] == station_id) & (im_obs_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='gray', linewidth=2.5, linestyle=(2, (2, 2)), label='Lee2022 observation')
ax.set_xlim(0.01, 10)
ax.set_ylim(0.00001, 1)
ax.set_xlabel('Vibration Period, T (s)', size=16)
ax.set_ylabel('Response Spectra, SA (g)', size=16)
ax.set_xscale('log')
ax.set_yscale('log')

ax = fig.add_subplot(gs[1, 0])
ax.text(0.012, 0.5, station_4, fontsize=14)
ax.text(0.012, 0.25, '$V_{S30}=%.0f~(m/s)$' % obs_df[obs_df['sta'] == station_4]['Vs30'].values[0], fontsize=14)
ax.text(0.012, 0.12, '$R_{rup}=%.0f~(km)$' % obs_df[(obs_df['sta'] == station_4) & (obs_df['evid'] == event)]['r_rup'].values[0], fontsize=14)
# Observation
ax.plot(periods, obs_df.loc[obs_df['sta'] == station_4, obs_df.filter(like='pSA').columns].to_numpy().flatten(), color='k', linewidth=2.5, label='Observation')
# Simulation NZVM v2.09
#sim_psa = sim_ims.pSA.sel(station=[station_4], component='rotd50')
sim_psa = sim_ims.pSA.sel(station=[station_4], component='geom')
ax.plot(sim_psa.period.values, sim_psa.values[0], color='r', linewidth=2.5, label='Simulation')
# Simulation NZVM v2.02
station_id = stations_K2026[stations_K2026['stat_name'] == station_4]['stat_id'].values[0]
ax.plot(periods_K2026, im_sim_K2026[(im_sim_K2026['stat_id'] == station_id) & (im_sim_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='g', linewidth=2.5, label='Lee2022 simulation (NZVM 2.02)')
# Observation Lee2022
ax.plot(periods_K2026, im_obs_K2026[(im_obs_K2026['stat_id'] == station_id) & (im_obs_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='gray', linewidth=2.5, linestyle=(2, (2, 2)), label='Lee2022 simulation (NZVM 2.02)')
ax.set_xlim(0.01, 10)
ax.set_ylim(0.00001, 1)
ax.set_xlabel('Vibration Period, T (s)', size=16)
ax.set_ylabel('Response Spectra, SA (g)', size=16)
ax.set_xscale('log')
ax.set_yscale('log')

ax = fig.add_subplot(gs[1, 1])
ax.text(0.012, 0.5, station_5, fontsize=14)
ax.text(0.012, 0.25, '$V_{S30}=%.0f~(m/s)$' % obs_df[obs_df['sta'] == station_5]['Vs30'].values[0], fontsize=14)
ax.text(0.012, 0.12, '$R_{rup}=%.0f~(km)$' % obs_df[(obs_df['sta'] == station_5) & (obs_df['evid'] == event)]['r_rup'].values[0], fontsize=14)
# Observation
ax.plot(periods, obs_df.loc[obs_df['sta'] == station_5, obs_df.filter(like='pSA').columns].to_numpy().flatten(), color='k', linewidth=2.5, label='Observation')
# Simulation NZVM v2.09
#sim_psa = sim_ims.pSA.sel(station=[station_5], component='rotd50')
sim_psa = sim_ims.pSA.sel(station=[station_5], component='geom')
ax.plot(sim_psa.period.values, sim_psa.values[0], color='r', linewidth=2.5, label='Simulation')
# Simulation NZVM v2.02
station_id = stations_K2026[stations_K2026['stat_name'] == station_5]['stat_id'].values[0]
ax.plot(periods_K2026, im_sim_K2026[(im_sim_K2026['stat_id'] == station_id) & (im_sim_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='g', linewidth=2.5, label='Lee2022 simulation (NZVM 2.02)')
# Observation
ax.plot(periods_K2026, im_obs_K2026[(im_obs_K2026['stat_id'] == station_id) & (im_obs_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='gray', linewidth=2.5, linestyle=(2, (2, 2)), label='Lee2022 observation')
ax.set_xlim(0.01, 10)
ax.set_ylim(0.00001, 1)
ax.set_xlabel('Vibration Period, T (s)', size=16)
ax.set_ylabel('Response Spectra, SA (g)', size=16)
ax.set_xscale('log')
ax.set_yscale('log')

ax = fig.add_subplot(gs[1, 2])
ax.text(0.012, 0.5, station_6, fontsize=14)
ax.text(0.012, 0.25, '$V_{S30}=%.0f~(m/s)$' % obs_df[obs_df['sta'] == station_6]['Vs30'].values[0], fontsize=14)
ax.text(0.012, 0.12, '$R_{rup}=%.0f~(km)$' % obs_df[(obs_df['sta'] == station_6) & (obs_df['evid'] == event)]['r_rup'].values[0], fontsize=14)
# Observation
ax.plot(periods, obs_df.loc[obs_df['sta'] == station_6, obs_df.filter(like='pSA').columns].to_numpy().flatten(), color='k', linewidth=2.5, label='Observation')
# Simulation NZVM v2.09
#sim_psa = sim_ims.pSA.sel(station=[station_6], component='rotd50')
sim_psa = sim_ims.pSA.sel(station=[station_6], component='geom')
ax.plot(sim_psa.period.values, sim_psa.values[0], color='r', linewidth=2.5, label='Simulation')
# Simulation NZVM v2.02
station_id = stations_K2026[stations_K2026['stat_name'] == station_6]['stat_id'].values[0]
ax.plot(periods_K2026, im_sim_K2026[(im_sim_K2026['stat_id'] == station_id) & (im_sim_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='g', linewidth=2.5, label='Lee2022 simulation (NZVM 2.02)')
# Observation
ax.plot(periods_K2026, im_obs_K2026[(im_obs_K2026['stat_id'] == station_id) & (im_obs_K2026['event_id'] == event_id)].filter(like='pSA').to_numpy().flatten(), color='gray', linewidth=2.5, linestyle=(2, (2,2)), label='Lee2022 simulation (NZVM 2.02)')
ax.set_xlim(0.01, 10)
ax.set_ylim(0.00001, 1)
ax.set_xlabel('Vibration Period, T (s)', size=16)
ax.set_ylabel('Response Spectra, SA (g)', size=16)
ax.set_xscale('log')
ax.set_yscale('log')

fig.subplots_adjust(top=0.94, bottom=0.05, left=0.04, right=0.99)

#plt.show()
plt.savefig('2013p544960.pdf', dpi=600)