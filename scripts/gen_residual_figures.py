"""
residualFigures
Created in 02-02-2022
Last modified on 31-10-2025
Felipe Kuncar (felipe.kuncar@canterbury.ac.nz | fkw22)
"""

# =====================================================================================================================
# IMPORT LIBRARIES AND FUNCTIONS
# =====================================================================================================================

import os.path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib
from pathlib import Path

# =====================================================================================================================
# DEFINE REFERENCE VARIABLES
# =====================================================================================================================

methodName1 = 'Finite-fault model'
methodName2 = 'Point-source model'

# =====================================================================================================================
# PATHS
# =====================================================================================================================

# Define the working directory
rootDir = Path(__file__).parent

# Define the folder that contains the inputs for the residual analysis
inputDir = Path(__file__).parent / 'residual_input_ff'
#inputDir = Path(__file__).parent / 'residual_input_ff_ps'

# Define the folder that contains the results of the residual analysis
#resultsDir = Path(__file__).parent / 'residual_output_ff'
resultsDir = Path(__file__).parent / 'residual_output_ff_ps'

''''
# =====================================================================================================================
# SITES
# =====================================================================================================================

version = 'v1'

computer = 1
if computer == 1:
    root = 'C:/Users/fkw21/OneDrive - University of Canterbury/PhD/7. RESEARCH/9. Publications/3. Journal Papers/'
elif computer == 2:
    root = 'C:/Users/Felipe Kuncar/Desktop/OneDrive - University of Canterbury/PhD/7. RESEARCH/9. Publications/3. Journal Papers/'

# Path to the root directory 1
rootDir1 = os.path.join(root, 'Paper3/Analysis/')

# Read site information
df_sites = pd.read_excel(os.path.join(rootDir1, 'versionControl.xlsx'), sheet_name='%s' % version)

# Extract site list
siteList = df_sites['Site ID'].tolist()
'''
# =====================================================================================================================
# READ OUTPUTS OF THE RESIDUAL ANALYSIS
# =====================================================================================================================

# Read output data as DataFrame
'''
events = pd.read_csv(os.path.join(inputDir, 'events.csv'))
stations = pd.read_csv(os.path.join(inputDir, 'stations.csv'))
bias_std = pd.read_csv(os.path.join(resultsDir, 'bias_std_df.csv'))
reStation = pd.read_csv(os.path.join(resultsDir, 'site_res_df.csv'))
reEvent = pd.read_csv(os.path.join(resultsDir, 'event_res_df.csv'))
'''

events = pd.read_csv(os.path.join(inputDir, 'events.csv'))
stations = pd.read_csv(os.path.join(inputDir, 'stations.csv'))

bias_std = pd.read_csv(os.path.join(resultsDir, 'ff', 'bias_std_df.csv'))
reStation = pd.read_csv(os.path.join(resultsDir, 'ff', 'site_res_df.csv'))
reEvent = pd.read_csv(os.path.join(resultsDir, 'ff', 'event_res_df.csv'))

#bias_std_p = pd.read_csv(os.path.join(resultsDir, 'ps', 'bias_std_df.csv'))

# Extract list of events
eventList = events.iloc[:, 1].to_list()

# Extract the vibration periods from the DataFrame bias_std
T = []
periods = bias_std.iloc[6:, 0]
for period in periods:
    period = period.replace('pSA_', '')
    period = float(period)
    T.append(period)

'''
# Extract the model bias and standard deviation values from the DataFrame bias_std
# Model bias, a, for SA
bias_SA = bias_std.loc[6:, 'bias']
# Model bias, a, for the other IMs
bias_IMs = bias_std.loc[0:5, 'bias']
# Between-event standard deviation, tau, for SA
tau_SA = bias_std.loc[6:, 'tau']
# Between-event standard deviation, tau, for other IMs
tau_IMs = bias_std.loc[0:5, 'tau']
# Site-to-site standard deviation, phiS2S, for SA
phiS2S_SA = bias_std.loc[6:, 'phi_S2S']
# Site-to-site standard deviation, phiS2S, for other IMs
phiS2S_IMs = bias_std.loc[0:5, 'phi_S2S']
# Within-event single-station standard deviation, phiSS, for SA
phiSS_SA = bias_std.loc[6:, 'phi_w']
# Within-event single-station standard deviation, phiSS, for other IMs
phiSS_IMs = bias_std.loc[0:5, 'phi_w']
# Total standard deviation, sigma, for SA
sigma_SA = bias_std.loc[6:, 'sigma']
# Total standard deviation, sigma, for the other IMs
sigma_IMs = bias_std.loc[0:5, 'sigma']
'''

# Model bias, a, for SA
bias_SA = bias_std.loc[6:, 'bias']
# Model bias, a, for the other IMs
bias_IMs = bias_std.loc[0:5, 'bias']
# Total standard deviation, sigma, for SA
sigma_SA = bias_std.loc[6:, 'sigma']
# Total standard deviation, sigma, for the other IMs
sigma_IMs = bias_std.loc[0:5, 'sigma']

'''
# Model bias, a, for SA
bias_SA_p = bias_std_p.loc[6:, 'bias']
# Model bias, a, for the other IMs
bias_IMs_p = bias_std_p.loc[0:5, 'bias']
# Total standard deviation, sigma, for SA
sigma_SA_p = bias_std_p.loc[6:, 'sigma']
# Total standard deviation, sigma, for the other IMs
sigma_IMs_p = bias_std_p.loc[0:5, 'sigma']
'''

'''
# Extract the between-event residual from the DataFrame reEvent
# Preallocate arrays that will contain the Be residuals
Be_SA = np.zeros((len(T), len(eventList)))
Be_IMs = np.zeros((6, len(eventList)))
for i, event in enumerate(eventList):
    event_id = events[events['event_name'] == event]['event_id'].values[0]
    reEvent.rename(columns={'Unnamed: 0': 'event'}, inplace=True)
    reEvent_event = reEvent[reEvent['event'] == 'event_' + str(event_id)]
    Be_SA[:, i] = reEvent_event.iloc[0, 7:]
    Be_IMs[:, i] = reEvent_event.iloc[0, 1:7]

# Extract the site-to-site residual from the DataFrame reStation
# Preallocate arrays that will contain the S2S residuals
S2S_SA = np.zeros((len(T), len(siteList)))
S2S_IMs = np.zeros((6, len(siteList)))
for i, site in enumerate(siteList):
    stat_id = stations[stations['stat_name'] == site]['stat_id'].values[0]
    reStation.rename(columns={'Unnamed: 0': 'station'}, inplace=True)
    reStation_site = reStation[reStation['station'] == 'stat_' + str(stat_id)]
    S2S_SA[:, i] = reStation_site.iloc[0, 7:]
    S2S_IMs[:, i] = reStation_site.iloc[0, 1:7]
'''

# =====================================================================================================================
# CREATE FIGURES
# =====================================================================================================================

# ---------------------------------------------------------------------------------------------------------------------
# 1) Model Bias and Total Standard Deviation
# ---------------------------------------------------------------------------------------------------------------------

# Create a figure
fig1 = plt.figure(figsize=(15, 5.5))

# Create the first gridspec for ploting the model biases
gs1 = fig1.add_gridspec(nrows=1, ncols=2, width_ratios=[3, 1], left=0.06, right=0.46, wspace=0.05)

# Model bias, a, for SA
ax1 = fig1.add_subplot(gs1[0, 0])
ax1.plot(T, bias_SA, 'r-', linewidth=2)
#ax1.plot(T, bias_SA_p, 'g-', linewidth=2)
ax1.plot([0.01, 10], [0, 0], color='0.4', linestyle=(0, (5, 5)), linewidth=2)
ax1.set_xscale('log')
ax1.set_xlim([0.01, 10])
ax1.set_ylim([-1.5, 1.5])
ax1.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax1.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
ax1.set_xlabel('Vibration Period, T (s)', size=14)
ax1.set_ylabel('Model Bias, $a$', size=14)
ax1.text(0.015, 1.30, "Underprediction", size=12)
ax1.text(0.015, -1.40, "Overprediction", size=12)

# Model bias, a, for other IMs
ax2 = fig1.add_subplot(gs1[0, 1])
x = [0, 1, 2, 3, 4, 5]
x_values = ["PGA", "PGV", "CAV", "AI", "$D_{s575}$", "$D_{s595}$"]
plt.xticks(x, x_values)
ax2.scatter(x_values, bias_IMs, s=40, c='r', marker='o', edgecolor='r')
#ax2.scatter(x_values, bias_IMs_p, s=40, c='g', marker='o', edgecolor='g')
ax2.plot([-1.0, 10.0], [0, 0], color='0.4', linestyle=(0, (5, 5)), linewidth=2)
ax2.set_xlim([-1.0, 6.0])
ax2.set_xticklabels(x_values, rotation=90)
ax2.set_ylim([-1.5, 1.5])
ax2.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax2.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# y-axis labels to the right side
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

# Create the second gridspec for ploting the standard deviations
gs2 = fig1.add_gridspec(nrows=1, ncols=2, width_ratios=[3, 1], left=0.57, right=0.97, wspace=0.05)

# Total standard deviation, sigma, for SA
ax3 = fig1.add_subplot(gs2[0, 0])
ax3.plot(T, sigma_SA, 'r-', label='%s' % methodName1, linewidth=2)
#ax3.plot(T, sigma_SA_p, 'g-', label='%s' % methodName2, linewidth=2)
ax3.legend(loc=3, fontsize=14)
ax3.set_xscale('log')
ax3.set_xlim([0.01, 10])
ax3.set_ylim([0.0, 1.0])
ax3.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax3.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
ax3.set_xlabel('Vibration Period, T (s)', size=14)
ax3.set_ylabel('Total Std. Dev., $\sigma$', size=14)
ax3.text(0.015, 1.40, "Underprediction", size=12)
ax3.text(0.015, -1.40, "Overprediction", size=12)

# Total standard deviation, sigma, for other IMs
ax4 = fig1.add_subplot(gs2[0, 1])
plt.xticks(x, x_values)
ax4.scatter(x_values, sigma_IMs, s=40, c='r', marker='o', edgecolor='r')
#ax4.scatter(x_values, sigma_IMs_p, s=40, c='g', marker='o', edgecolor='g')
# ax4.plot([-1.0, 7.0], [0, 0], 'k--', linewidth=1)
ax4.set_xlim([-1.0, 6.0])
ax4.set_xticklabels(x_values, rotation=90)
ax4.set_ylim([0.0, 1.0])
ax4.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax4.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# y-axis labels to the right side
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()

fig1.subplots_adjust(top=0.98, bottom=0.10, left=0.25, right=0.98)

# Save figure
# plt.savefig('model_bias.pdf', dpi=300)
plt.savefig(os.path.join(resultsDir, 'modelBias.pdf'))
plt.savefig(os.path.join(resultsDir, 'modelBias.png'), dpi=600)

'''
# ---------------------------------------------------------------------------------------------------------------------
# 2) Standard Deviations
# ---------------------------------------------------------------------------------------------------------------------

# Create a figure
fig2 = plt.figure(figsize=(6, 10))

# create the gridspec for ploting the total standard deviations
gs1 = fig2.add_gridspec(nrows=4, ncols=2, width_ratios=[4, 1], wspace=0.05, hspace=0.40)

# total standard deviation, sigma, for SA
ax1 = fig2.add_subplot(gs1[0, 0])
ax1.plot(T, sigma_SA, 'b-', linewidth=1.5)
ax1.set_xscale('log')
ax1.set_xlim([0.01, 10])
ax1.set_ylim([0.0, 1.0])
ax1.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax1.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# ax1.set_xlabel('Vibration Period, T (s)', size=12)
ax1.set_ylabel('$\sigma$', size=12)

# Total standard deviation, sigma, for other IMs
ax2 = fig2.add_subplot(gs1[0, 1])
plt.xticks(x, x_values)
ax2.scatter(x_values, sigma_IMs, s=30, c='b', marker='o', edgecolor='b')
ax2.set_xlim([-1.0, 6.0])
ax2.set_xticklabels(x_values, rotation=90)
ax2.set_ylim([0.0, 1.0])
ax2.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax2.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# y-axis labels to the right side
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

# Between-event standard deviation, tau, for SA
ax3 = fig2.add_subplot(gs1[1, 0])
ax3.plot(T, tau_SA, 'b-', linewidth=1.5)
ax3.set_xscale('log')
ax3.set_xlim([0.01, 10])
ax3.set_ylim([0.0, 0.9])
ax3.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax3.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# ax3.set_xlabel('Vibration Period, T (s)', size=12)
ax3.set_ylabel('$\u03C4$', size=14)

# Between-event standard deviation, tau, for other IMs
ax4 = fig2.add_subplot(gs1[1, 1])
plt.xticks(x, x_values)
ax4.scatter(x_values, tau_IMs, s=30, c='b', marker='o', edgecolor='b')
ax4.set_xlim([-1.0, 6.0])
ax4.set_xticklabels(x_values, rotation=90)
ax4.set_ylim([0.0, 0.9])
ax4.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax4.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# y-axis labels to the right side
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()

# Site-to-site standard deviation, phiS2S, for SA
ax5 = fig2.add_subplot(gs1[2, 0])
ax5.plot(T, phiS2S_SA, 'b-', linewidth=1.5)
ax5.set_xscale('log')
ax5.set_xlim([0.01, 10])
ax5.set_ylim([0.0, 0.5])
ax5.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax5.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# ax5.set_xlabel('Vibration Period, T (s)', size=12)
ax5.set_ylabel('$\phi_{S2S}$', size=14)

# Site-to-site standard deviation, phiS2S, for other IMs
ax6 = fig2.add_subplot(gs1[2, 1])
plt.xticks(x, x_values)
ax6.scatter(x_values, phiS2S_IMs, s=30, c='b', marker='o', edgecolor='b')
ax6.set_xlim([-1.0, 6.0])
ax6.set_xticklabels(x_values, rotation=90)
ax6.set_ylim([0.0, 0.5])
ax6.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax6.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# y-axis labels to the right side
ax6.yaxis.set_label_position("right")
ax6.yaxis.tick_right()

# Within-event single-station standard deviation, phiSS, for SA
ax7 = fig2.add_subplot(gs1[3, 0])
ax7.plot(T, phiSS_SA, 'b-', label='%s' % methodName, linewidth=1.5)
ax7.legend(loc=3)
ax7.set_xscale('log')
ax7.set_xlim([0.01, 10])
ax7.set_ylim([0.0, 0.5])
ax7.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax7.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
ax7.set_xlabel('Vibration Period, T (s)', size=12)
ax7.set_ylabel('$\phi_{SS}$', size=12)

# Within-event single-station standard deviation, phiSS, for other IMs
ax8 = fig2.add_subplot(gs1[3, 1])
plt.xticks(x, x_values)
ax8.scatter(x_values, phiSS_IMs, s=35, c='b', marker='o', edgecolor='b')
ax8.set_xlim([-1.0, 6.0])
ax8.set_xticklabels(x_values, rotation=90)
ax8.set_ylim([0.0, 0.5])
ax8.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax8.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# y-axis labels to the right side
ax8.yaxis.set_label_position("right")
ax8.yaxis.tick_right()

# Save figure
# plt.savefig('std_devs.pdf', dpi=300)
plt.savefig(os.path.join(resultsDir, 'stdDevs.pdf'), dpi=300)

# ---------------------------------------------------------------------------------------------------------------------
# 2) Between-Event Residuals
# ---------------------------------------------------------------------------------------------------------------------

# Create a figure
fig3 = plt.figure(figsize=(8, 5))

# create the gridspec for ploting the site-to-site residuals
gs1 = fig3.add_gridspec(nrows=1, ncols=2, width_ratios=[4, 1], wspace=0.05, hspace=0.40)

# SA
ax1 = fig3.add_subplot(gs1[0, 0])
ax1.set_xlabel('Vibration Period, T (s)', size=12)
ax1.set_xscale('log')
ax1.set_xlim([0.01, 10])
ax1.set_ylim([-2.0, 2.0])
ax1.text(0.015, 1.80, "Underprediction", size=12)
ax1.text(0.015, -1.85, "Overprediction", size=12)
ax1.text(0.7, 1.80, "%s" % methodName, size=10)
# ax1.set_xlabel('Vibration Period, T (s)', size=12)
ax1.set_ylabel('Between-Event Residual, $\delta B_e$', size=12)
for i, event in enumerate(eventList):
    if event == '3468622':
        ax1.plot(T, Be_SA[:, i], color='g', linewidth=1.5, label='3468622')
    else:
        ax1.plot(T, Be_SA[:, i], color='gray', linewidth=0.6)
ax1.legend(loc=4, fontsize=8)
ax1.plot([-1.0, 10.0], [0, 0], color='k', linewidth=2)
ax1.plot(T, tau_SA, color='k', linestyle=(0, (5, 5)), linewidth=2.0)
ax1.plot(T, -tau_SA, color='k', linestyle=(0, (5, 5)), linewidth=2.0)

# Other IMs
ax2 = fig3.add_subplot(gs1[0, 1])
plt.xticks(x, x_values)
ax2.set_xlim([-1.0, 6.0])
ax2.set_xticklabels(x_values, rotation=90)
ax2.set_ylim([-2.0, 2.0])
ax2.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax2.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# y-axis labels to the right side
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
for i, event in enumerate(eventList):
    if event == '3468622':
        ax2.scatter(x_values, Be_IMs[:, i], s=20, color='g', marker='o', edgecolor='g', label='3468622')
    else:
        ax2.scatter(x_values, Be_IMs[:, i], s=10, color='gray', marker='o', edgecolor='gray')
ax2.scatter(x_values, [0, 0, 0, 0, 0, 0], s=20, color='k', marker='o')
ax2.scatter(x_values, tau_IMs, s=50, color='k', marker='_')
ax2.scatter(x_values, -tau_IMs, s=50, color='k', marker='_')

# Save figure
# plt.savefig('S2S_residual_all.pdf', dpi=300)
plt.savefig(os.path.join(resultsDir, 'BeResidualAll.pdf'), dpi=300)

# ---------------------------------------------------------------------------------------------------------------------
# 4) Site-to-Site Residuals
# ---------------------------------------------------------------------------------------------------------------------

# Create a figure
fig4 = plt.figure(figsize=(8, 5))

# create the gridspec for ploting the site-to-site residuals
gs1 = fig4.add_gridspec(nrows=1, ncols=2, width_ratios=[4, 1], wspace=0.05, hspace=0.40)

# SA
ax1 = fig4.add_subplot(gs1[0, 0])
ax1.plot([-1.0, 10.0], [0, 0], color='0.4', linestyle=(0, (5, 5)), linewidth=2)
ax1.set_xlabel('Vibration Period, T (s)', size=12)
ax1.set_xscale('log')
ax1.set_xlim([0.01, 10])
ax1.set_ylim([-1.0, 1.0])
ax1.text(0.015, 0.80, "Underprediction", size=12)
ax1.text(0.015, -0.80, "Overprediction", size=12)
ax1.text(0.7, 0.80, "%s" % methodName, size=10)
# ax1.set_xlabel('Vibration Period, T (s)', size=12)
ax1.set_ylabel('Site-to-Site Residual, $\delta S2S_s$', size=12)
i = 0
for i, site in enumerate(siteList):
    ax1.plot(T, S2S_SA[:, i], color='0.6', linewidth=0.6)
    i = i + 1
ax1.plot([-1.0, 10.0], [0, 0], color='k', linewidth=2.0)
ax1.plot(T, phiS2S_SA, color='k', linestyle=(0, (5, 5)), linewidth=2.0)
ax1.plot(T, -phiS2S_SA, color='k', linestyle=(0, (5, 5)), linewidth=2.0)

# Other IMs
ax2 = fig4.add_subplot(gs1[0, 1])
plt.xticks(x, x_values)
ax2.set_xlim([-1.0, 6.0])
ax2.set_xticklabels(x_values, rotation=90)
ax2.set_ylim([-1.0, 1.0])
ax2.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
ax2.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
# y-axis labels to the right side
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
for i, site in enumerate(siteList):
    ax2.scatter(x_values, S2S_IMs[:, i], s=10, color='0.6', marker='o', edgecolor='0.6')
ax2.scatter(x_values, [0, 0, 0, 0, 0, 0], s=20, color='k', marker='o')
ax2.scatter(x_values, phiS2S_IMs, s=50, color='k', marker='_')
ax2.scatter(x_values, -phiS2S_IMs, s=50, color='k', marker='_')

# Save figure
# plt.savefig('S2S_residual_all.pdf', dpi=300)
plt.savefig(os.path.join(resultsDir, 'S2SResidualAll.pdf'), dpi=300)

plt.close('all')

# ---------------------------------------------------------------------------------------------------------------------
# 5) Systematic Residuals
# ---------------------------------------------------------------------------------------------------------------------

# Loop over the sites
for i, site in enumerate(siteList):

    # Create a figure
    fig5 = plt.figure(i, figsize=(8, 5))

    # Create a gridspec
    gs1 = fig5.add_gridspec(nrows=1, ncols=2,  width_ratios=[4, 1], wspace=0.05)

    # Systematic residual for SA
    ax1 = fig5.add_subplot(gs1[0, 0])
    ax1.plot(T, bias_SA + S2S_SA[:, i], 'b-', label='%s' % methodName, linewidth=2)
    ax1.legend(loc=4)
    ax1.plot([0.01, 10], [0, 0], color='0.4', linestyle=(0, (5, 5)), linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlim([0.01, 10])
    ax1.set_ylim([-1.25, 1.25])
    ax1.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
    ax1.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
    ax1.set_xlabel('Vibration Period, T (s)', size=14)
    ax1.set_ylabel('Systematic Residual, $a + \delta S2S_s$', size=14)
    ax1.text(3.0, 1.05, "%s" % site, size=16)
    ax1.text(0.015, 0.90, "Underprediction", size=12)
    ax1.text(0.015, -0.95, "Overprediction", size=12)

    # Systematic residual for other IMs
    ax2 = fig5.add_subplot(gs1[0, 1])
    x = [0, 1, 2, 3, 4, 5]
    x_values = ["PGA", "PGV", "CAV", "AI", "$D_{s575}$", "$D_{s595}$"]
    plt.xticks(x, x_values)
    ax2.scatter(x_values, bias_IMs + S2S_IMs[:, i], s=40, c='b', marker='o', edgecolor='b')
    ax2.plot([-1.0, 10.0], [0, 0], color='0.4', linestyle=(0, (5, 5)), linewidth=2)
    ax2.set_xlim([-1.0, 6.0])
    ax2.set_xticklabels(x_values, rotation=90)
    ax2.set_ylim([-1.25, 1.25])
    ax2.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.2)
    ax2.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.2)
    # y-axis labels to the right side
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

# Save figures
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(resultsDir, 'systematicResiduals.pdf'))
for fig in range(0, len(siteList)):
    pdf.savefig(fig)

pdf.close()
plt.close('all')
'''