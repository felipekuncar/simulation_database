"""
residualAnalysis
Created on 13-10-2022
Last modified on 31-10-2025
Created by Claudio Schill (claudio.schill@canterbury.ac.nz)
Modifed by Felipe Kuncar (felipe.kuncar@canterbury.ac.nz | fkw22)
"""

# =====================================================================================================================
# IMPORT LIBRARIES AND FUNCTIONS
# =====================================================================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import mera
from mera import run_mera, MeraResults
import sys
print(sys.executable)

# =====================================================================================================================
# LOAD THE DATA
# =====================================================================================================================

data_dir = Path(__file__).parent / "residual_input_ff"
output_dir = Path(__file__).parent / "residual_output_ff"
#data_dir = Path(__file__).parent / "residual_input_ff_ps"
#output_dir = Path(__file__).parent / "residual_output_ff_ps"
output_dir.mkdir(exist_ok=True)

stations_ffp = data_dir / "stations.csv"
events_ffp = data_dir / "events.csv"
obs_ffp = data_dir / "im_obs.csv"
sim_ffp = data_dir / "im_sim.csv"
#sim_ffp = data_dir / "im_sim_p.csv"

stations_df = pd.read_csv(stations_ffp, index_col=0)
events_df = pd.read_csv(events_ffp, index_col=0)

obs_df = pd.read_csv(obs_ffp, index_col=0)
sim_df = pd.read_csv(sim_ffp, index_col=0)

# Sanity checking
# Ensure the index matches
assert (
    np.all(obs_df.index == sim_df.index)
    and np.all(obs_df.event_id == sim_df.event_id)
    and np.all(obs_df.stat_id == sim_df.stat_id)
)

'''
# List of IMs of interest
# Just using all shared pSA periods in the data files
ims = [
    cur_im
    for cur_im in np.intersect1d(obs_df.columns, sim_df.columns)
    if cur_im.startswith("pSA")
]
'''

# =====================================================================================================================
# PERFORM RESIDUAL ANALYSIS
# =====================================================================================================================

# Select intensity measures
ims = list(obs_df.columns.values)
ims.pop(0)
ims.pop(0)

# Compute the residual
res_df = (obs_df[ims] / sim_df[ims]).apply(np.log)

# Add event id and station id columns
res_df["event_id"] = np.char.add("event_", obs_df.event_id.values.astype(str))
res_df["stat_id"] = np.char.add("stat_", obs_df.stat_id.values.astype(str))

# Run MER
mera_result = run_mera(
    res_df, list(ims), "event_id", "stat_id"
)

# Save the results
mera_result.save(output_dir)
loaded = mera.MeraResults.load(output_dir)

# mera_result.event_res_df.to_csv(output_dir / "event_residuals.csv", index_label="event_id")
# mera_result.site_res_df.to_csv(output_dir / "site_residuals.csv", index_label="stat_id")
# mera_result.rem_res_df.to_csv(output_dir / "remaining_residuals.csv", index_label="gm_id")
# mera_result.bias_std_df.to_csv(output_dir / "bias_std.csv", index_label="IM")