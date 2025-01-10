import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

figwidth = 597/72
fontsize = 12
labelpad = -3

def red_green_red_cmap():
    """
    Create a custom colormap that is green at 0 and red at both ends.
    """
    colors = [(0, 'red'), (0.5, 'green'), (1, 'red')]  # Red at both ends, green at the middle
    cmap = LinearSegmentedColormap.from_list("RedGreenRed", colors, N=256)
    return cmap

def plotSNR(csv_filename="SNR_comparison.csv"):
    """
    Plot error data for different SNR values from CSV.
    """
    # Load data from CSV as a structured array
    data = np.genfromtxt(
        f"data/{csv_filename}", delimiter=",", dtype=None, encoding="utf-8", names=True
    )

    # Define parameters and colors with more descriptive names
    parameters = [ "pvs_axial", "pvs_radial", "alpha", "pvs_fraction"]
    parameter_labels = {
        "pvs_axial": "PVS Axial Diffusivity (mm²/s × 10⁻³)",
        "pvs_radial": "PVS Radial Diffusivity (mm²/s × 10⁻³)",
        "alpha": "Angle Between PVS and WM (°)",
        "pvs_fraction": "PVS Volume Fraction"
    }
    color_palette = ["red", "blue", "green"]  # Colors for different SNR values

    # Prepare the plot
    fig, axes = plt.subplots(2, 2, figsize=(figwidth, figwidth * 1.1))
    axes = axes.flatten()

    # List to hold all legend handles and labels
    handles, labels = [], []

    for i, param in enumerate(parameters):
        ax = axes[i]

        # Filter data for the current parameter
        param_data = data[data["Parameter"] == param]

        if param == "alpha":
            param_data["Value"] = np.degrees(param_data["Value"].astype(float))        
        
        # Multiply axial and radial diffusivity by 1000
        if param == "pvs_axial" or param == "pvs_radial":
            param_data["Value"] = param_data["Value"].astype(float) * 1000
            
        ax.set_xlim([np.min(param_data["Value"]), np.max(param_data["Value"])])
        ax.set_ylim([np.min(param_data["Error"]), np.max(param_data["Error"])])


        # Loop through unique SNR values
        for j, SNR in enumerate(np.unique(param_data["SNR"])):
            SNR_data = param_data[param_data["SNR"] == SNR]

            x_values = SNR_data["Value"].astype(float)
            y_values = SNR_data["Error"].astype(float)

            line, = ax.plot(
                x_values,
                y_values,
                color=color_palette[j % len(color_palette)],
                label=f"SNR {SNR}",
            )


        fontplus = 2
        # Use more descriptive names for the axis labels
        ax.set_xlabel(parameter_labels[param], fontsize=11 + fontplus, labelpad=2)
        ax.set_ylabel("% Error", fontsize=11 + fontplus, labelpad=2)

         # Increase tick font size
        ax.tick_params(axis='both', which='major', labelsize=9 + fontplus)  # Increase major ticks font size
        ax.tick_params(axis='both', which='minor', labelsize=9 + fontplus)  # Optionally, increase minor ticks font size

        if param == "pvs_axial":
            ax.legend(fontsize=14)  

    # Adjust layout
    plt.tight_layout(h_pad=2)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    # Extract the base name from the CSV file and create an output file name
    base_filename = os.path.splitext(os.path.basename(csv_filename))[0]
    output_filename = f"figures/{base_filename}.pdf"

    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}.")

def plotDiffusivities(csv_filename="PVS_diffusivity_errors_all.csv"):
    """
    Plot 3D surface plots for errors from CSV data.
    """
    # Load data from CSV
    data = np.genfromtxt(f"data/{csv_filename}", delimiter=",", skip_header=1)
    axial_values = data[:, 0]
    radial_values = data[:, 1]
    errors1 = data[:, 2]
    errors2 = data[:, 3]
    errors3 = data[:, 4]

    # Create grid for plotting
    axial_unique = np.unique(axial_values)
    radial_unique = np.unique(radial_values)

    grid_shape = (len(axial_unique), len(radial_unique))
    errors_reshaped1 = np.full(grid_shape, np.nan)
    errors_reshaped2 = np.full(grid_shape, np.nan)
    errors_reshaped3 = np.full(grid_shape, np.nan)

    for axial, radial, err1, err2, err3 in zip(axial_values, radial_values, errors1, errors2, errors3):
        ax_idx = np.where(axial_unique == axial)[0][0]
        rad_idx = np.where(radial_unique == radial)[0][0]
        errors_reshaped1[ax_idx, rad_idx] = err1
        errors_reshaped2[ax_idx, rad_idx] = err2
        errors_reshaped3[ax_idx, rad_idx] = err3
    
    # Function to handle outliers by clipping for color normalization only
    def clip_for_color(errors, clip_factor=3):
        """
        Clip extreme values based on standard deviation for color normalization only.
        Values outside [mean - clip_factor * std, mean + clip_factor * std] will be clipped.
        """
        mean = np.nanmean(errors)
        std = np.nanstd(errors)
        lower_bound = mean - clip_factor * std
        upper_bound = mean + clip_factor * std
        return np.clip(errors, lower_bound, upper_bound)

    # Apply clipping only for color normalization
    clipped_errors1 = clip_for_color(errors_reshaped1)
    clipped_errors2 = clip_for_color(errors_reshaped2)
    clipped_errors3 = clip_for_color(errors_reshaped3)

    # Plotting helper function
    def plot_surface(errors, clipped_errors, title, filename, axial_unique, radial_unique):
        fig = plt.figure(figsize=(figwidth/ 2, figwidth / 2))
        ax = fig.add_subplot(111, projection="3d")
        Axial, Radial = np.meshgrid(axial_unique, radial_unique, indexing="ij")
        
        # Create the custom colormap
        cmap = red_green_red_cmap()

        # Normalize: center at 0, scale to the maximum deviation from zero (using clipped data for color scale)
        norm = Normalize(vmin=-np.nanmax(np.abs(clipped_errors)), vmax=np.nanmax(np.abs(clipped_errors)))

        # Plot surface with custom colormap and normalization
        surface = ax.plot_surface(Axial * 1000, Radial * 1000, errors, cmap=cmap, norm=norm)

        # Set labels and title
        ax.set_xlabel("λ‖ (mm²/s × 10⁻³)", fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel("λ⊥ (mm²/s × 10⁻³)", fontsize=fontsize, labelpad=labelpad)
        ax.set_zlabel("Error (%)", fontsize=fontsize, labelpad=labelpad)
        # ax.set_title(title)

        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2, pad=-2)  # Increase tick label size for both axes
        ax.tick_params(axis='z', which='major', labelsize=fontsize-2, pad=-1)      # Increase tick label size for z-axis
        
        # Add color bar
        # fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.9, top=1, bottom=0.1)

        
        # Save the plot
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}.")

    # Generate plots
    plot_surface(errors_reshaped1, clipped_errors1, "Error in ADC along PVS Estimation (Single Tensor)", "figures/PVSdiffusivities_model1.pdf", axial_unique, radial_unique)
    plot_surface(errors_reshaped2, clipped_errors2, "Error in ADC along PVS Estimation (Multi Tensor)", "figures/PVSdiffusivities_model2.pdf", axial_unique, radial_unique)
    plot_surface(errors_reshaped3, clipped_errors3, "Error in ADC along PVS Estimation (Multi Tensor Constrained)", "figures/PVSdiffusivities_model3.pdf", axial_unique, radial_unique)

def plotAlphaVolume(csv_filename="PVS_alpha_volume_errors.csv"):
    """
    Plot 3D surface plots for errors from CSV data, with smoothing applied.
    """
    # Load data from CSV
    data = np.genfromtxt(f"data/{csv_filename}", delimiter=",", skip_header=1)
    alpha_values = np.rad2deg(data[:, 0])
    volume_fraction_values = data[:, 1]
    errors1 = data[:, 2]
    errors2 = data[:, 3]
    errors3 = data[:, 4]

    # Create grid for plotting
    alpha_unique = np.unique(alpha_values)
    volume_fraction_unique = np.unique(volume_fraction_values)

    grid_shape = (len(alpha_unique), len(volume_fraction_unique))
    errors_reshaped1 = np.full(grid_shape, np.nan)
    errors_reshaped2 = np.full(grid_shape, np.nan)
    errors_reshaped3 = np.full(grid_shape, np.nan)

    for alpha, volume_fraction, err1, err2, err3 in zip(alpha_values, volume_fraction_values, errors1, errors2, errors3):
        alpha_idx = np.where(alpha_unique == alpha)[0][0]
        vf_idx = np.where(volume_fraction_unique == volume_fraction)[0][0]
        errors_reshaped1[alpha_idx, vf_idx] = err1
        errors_reshaped2[alpha_idx, vf_idx] = err2
        errors_reshaped3[alpha_idx, vf_idx] = err3

    # Function to handle outliers by clipping for color normalization only
    def clip_for_color(errors, clip_factor=3):
        """
        Clip extreme values based on standard deviation for color normalization only.
        Values outside [mean - clip_factor * std, mean + clip_factor * std] will be clipped.
        """
        mean = np.nanmean(errors)
        std = np.nanstd(errors)
        lower_bound = mean - clip_factor * std
        upper_bound = mean + clip_factor * std
        return np.clip(errors, lower_bound, upper_bound)

    # Apply clipping only for color normalization
    clipped_errors1 = clip_for_color(errors_reshaped1)
    clipped_errors2 = clip_for_color(errors_reshaped2)
    clipped_errors3 = clip_for_color(errors_reshaped3)

    # Plotting helper function
    def plot_surface(errors, clipped_errors, title, filename):
        fig = plt.figure(figsize=(figwidth/ 2, figwidth / 2))
        ax = fig.add_subplot(111, projection="3d")
        Alpha, VolumeFraction = np.meshgrid(alpha_unique, volume_fraction_unique, indexing="ij")
        
        # Create the custom colormap
        cmap = red_green_red_cmap()

        # Normalize: center at 0, scale to the maximum deviation from zero (using clipped data for color scale)
        norm = Normalize(vmin=-np.nanmax(np.abs(clipped_errors)), vmax=np.nanmax(np.abs(clipped_errors)))

        # Plot surface with custom colormap and normalization
        surface = ax.plot_surface(Alpha, VolumeFraction, errors, cmap=cmap, norm=norm)
        
        
        # Set labels and title
        ax.set_xlabel("Alpha (°)", fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel("PVS VF", fontsize=fontsize, labelpad=labelpad)
        ax.set_zlabel("Error (%)", fontsize=fontsize, labelpad=labelpad+ 1)
        # ax.set_title(title)

        # Increase tick label size
        ax.tick_params(axis='both', which='major', labelsize=fontsize-1, pad=-2)  # Increase tick label size for both axes
        ax.tick_params(axis='z', which='major', labelsize=fontsize-1, pad=0)      # Increase tick label size for z-axis

        # Add color bar
        # fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.9, top=1, bottom=0.1)

        # Save the plot
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}.")

    # Generate plots with smoothed data
    plot_surface(errors_reshaped1, clipped_errors1, "Error in ADC along PVS Estimation (Single Tensor)", "figures/AlphaVolume_model1.pdf")
    plot_surface(errors_reshaped2, clipped_errors2, "Error in ADC along PVS Estimation (Multi Tensor)", "figures/AlphaVolume_model2.pdf")
    plot_surface(errors_reshaped3, clipped_errors3, "Error in ADC along PVS Estimation (Multi Tensor Constrained)", "figures/AlphaVolume_model3.pdf")

plotSNR("SNR_comparison_SingleTensor.csv")
plotSNR("SNR_comparison_MultiTensor.csv")
plotSNR("SNR_comparison_MultiTensorConstrained.csv")

plotDiffusivities()
plotAlphaVolume()

plt.show()
