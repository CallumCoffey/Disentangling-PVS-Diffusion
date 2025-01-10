import nibabel as nib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sh2Peak_WM     = 'Pipeline/output/HCP_1200/167440/disentangling/sh2Peak_WM.nii'
sh2Peak_PVS    = 'Pipeline/output/HCP_1200/167440/disentangling/sh2Peak_PVS.nii'
sh2Peak_stats  = 'Pipeline/output/HCP_1200/167440/disentangling/sh2Peak_stats.txt'

def compute_stats(file_path, max_peaks=5):
    """
    Load the NIfTI file, reshape it to separate peaks, compute vector magnitudes,
    and calculate basic statistics (max, min, mean, std, non-zero count) for each peak.
    """
    # Load NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()

    # Reshape the fourth dimension to separate peaks (n_peaks = dim_4 / 3)
    n_voxels = data.shape[3]
    if n_voxels % 3 != 0:
        raise ValueError(f"Fourth dimension ({n_voxels}) is not divisible by 3.")
    
    n_peaks = n_voxels // 3
    reshaped = data.reshape(*data.shape[:3], n_peaks, 3)

    print('reshaped = ', reshaped.shape)

    # Calculate magnitudes for each peak separately
    peak_magnitudes = [np.linalg.norm(reshaped[:,:,:,i,:], axis=-1) for i in range(n_peaks)]

    # Initialize list to store statistics for each peak
    peak_stats = []
    
    # For each peak, calculate the statistics across all non-zero voxels
    for i in range(min(max_peaks, n_peaks)):  # Up to 5 peaks or fewer if n_peaks < 5
        # Filter out NaN and zero values
        peak_values = peak_magnitudes[i]
        peak_values = peak_values[~np.isnan(peak_values)]  # Remove NaNs
        peak_values = peak_values[peak_values != 0]  # Remove zeros

        # Calculate statistics for this peak
        peak_stat = {
            'max': np.max(peak_values),
            'min': np.min(peak_values),
            'mean': np.mean(peak_values),
            'std': np.std(peak_values),
            'non_zero_count': len(peak_values),  # Count of non-zero voxels for this peak
            'magnitudes': peak_values  # Store the magnitudes for violin plotting
        }
        peak_stats.append(peak_stat)

    return peak_stats


# Compute stats for WM peaks
peak_stats_WM = compute_stats(sh2Peak_WM)

# Compute stats for PVS peaks
peak_stats_PVS = compute_stats(sh2Peak_PVS)

# Prepare data for violin plot
def prepare_violin_data(peak_stats_PVS, peak_stats_WM, max_peaks=5):
    """
    Prepare the data for plotting violin plots by collecting magnitudes of each peak.
    """
    data = []
    # Add PVS peak
    pvs_magnitudes = peak_stats_PVS[0]['magnitudes']
    data.extend([{'Peak': 'PVS', 'Magnitude': mag} for mag in pvs_magnitudes])

    # Add WM peaks (up to 5)
    for i in range(min(max_peaks, len(peak_stats_WM))):
        wm_magnitudes = peak_stats_WM[i]['magnitudes']
        data.extend([{'Peak': f'WM_Peak_{i+1}', 'Magnitude': mag} for mag in wm_magnitudes])
    
    # Convert to DataFrame for seaborn
    df = pd.DataFrame(data)
    return df

# Prepare data for violin plot
df = prepare_violin_data(peak_stats_PVS, peak_stats_WM)

figwidth = 597/72

# Plot violin plots
plt.figure(figsize=(figwidth, figwidth*0.8))  # Increase the figure size to accommodate all peaks
sns.violinplot(x='Peak', y='Magnitude', data=df, scale='width', inner="quart")
plt.title('Distribution of Peak Magnitudes for PVS and WM (All Peaks)')
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()

# Write stats to a file in table format
with open(sh2Peak_stats, 'w') as f:
    # Write header with proper alignment
    f.write(f"{'Statistic':<20}{'PVS':<15}{'Largest_Peak':<20}{'Second_Largest_Peak':<20}{'Third_Largest_Peak':<20}{'Fourth_Largest_Peak':<20}{'Fifth_Largest_Peak':<20}\n")
    
    # Write the stats for each parameter in separate rows
    for key in ['max', 'min', 'mean', 'std', 'non_zero_count']:
        f.write(f"{key:<20}{peak_stats_PVS[0][key]:<15.4f}")
        for peak_stat in peak_stats_WM:
            f.write(f"{peak_stat[key]:<20.4f}")
        f.write("\n")
