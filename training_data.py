"""The script for generating 3D TPMDs, and saving the slices as training data."""

# Create conda environment for MCM first. In the terminal, run:
# conda env create -f environment.yml
from MCM_functions_new import *



#############################################################################
# Set random seed for reproducibility
SEED = 42  
np.random.seed(SEED)
random.seed(SEED)
rng = np.random.RandomState(SEED)           # For sampling random slices each time it is run

# If you use scikit-learn or other libraries that use randomness:
try:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

try:
    os.environ['PYTHONHASHSEED'] = str(SEED)
except Exception:
    pass

#############################################################################
"""Defining Training Data Parameters"""
n_samples = 5                                                              # Number of 3D TPMD samples to generate – 2100 used for training data                                     
sigma = 3    
#############################################################################
"""Defining Modified Cormack Method Parameters"""

N = 513                                                                     # image size (N x N pixels) of copper projections                                    
raw_proj_dir = "Data_Generation_Required/Cu_20Projections(513)"           # directory containing ideal copper projections
# raw_proj_dir = "Cu_20Projections(513)"

pang = np.linspace(0, 45, 20)                                               # projection angles for the 20 projections for the DFT copper data        
ncoeffs = [150, 120, 100, 90, 80, 50]                                       # number of Chebyshev coefficients for each projection angle      
ncoeffs.extend([30]* (20-len(ncoeffs)))

order = 4                                                                   # copper's FCC structure means it has 4-fold symmetry (order = 4)                                  
nphi = 180                                                                  # angle increments for delta in polar unit circle are given by 90/nphi; 'm_max = nphi' when computing Chebyshev coefficients      
# calib = 0.09808                                                             # Experimental data calibration factor (pixels per mm) for copper data — NOT USED                           
xsize = N // 2                                                              # Centre of the image in pixels
rhocut = 1                                                                  # 1 = don't cut off, 0 = cut off at flvl 
flvl = 100                                                                  # Choose Fermi level so the cutoff starts at your desired px (e.g., 256) (max value is N/2 = 256 for N=513)
kt = 6.0                                                                    # Slope; increase for smoother, decrease for sharper
rhofn = [rhocut, flvl, kt]
rhofn_PCA = [0, 200, 3.0]

# Generating Experimental Data.
# num_simulations = 1                                                         # Number of simulated 3D TPMD datasets to generate                               
count_ttl = [200_000_000, 
             100_000_000, 
             50_000_000, 
             10_000_000, 
             1_000_000]                                                     # What should the simulated total counts be? ≈200,000,000 corresponds to 3 months of measurements!


base_dir = "Data_Generation_Required/TPMD_Data"
# base_dir = "/store/LION/gfbl2/DeepCormack_Data"


# # Directory to save side-by-side images
# tpmd_image_dir = os.path.join(base_dir_meas, "TPMD_Image")
# os.makedirs(tpmd_image_dir, exist_ok=True)
#############################################################################

""" --- Generating High-Quality TPMD Central Slices using PCA on Copper 3D TPMD ---
Calculate a_n^m and rho_n for the ideal DFT copper projections. PCA per channel 'n' will be used on copper a_n^m 
coefficients all 256 slices combined. Randomly sampling from each of the 'n' the latent spaces will generate new, 
realistic a_n^m coefficients, giving us a consistent way to create new, realistic rho_n for TPMD central slices.

    Parameters
    ----------
    raw_projs : shape (N, N, nproj) = (513, 513, 20)
    anm_Cu : shape (xsize, nphi, nproj) = (256, 180, 20)
    rhoreturn_ideal_Cu : shape (xsize, nphi, xsize) = (256, 20, 256). 
                            The final index denotes the rho_n slices.
"""

# Fetching the ideal copper projections
file_names = [f"I_TPMD2D.OUT_PROJ_{i+1}" for i in range(20)]
raw_projs = load_projections(raw_proj_dir, file_names, N)                   # shape (N, N, 20)
y_max_Cu = raw_projs.shape[0]//2                                            # For real data, we have to assume
rhoreturn_ideal_Cu, anm_Cu = getrho(raw_projs, order, pang, nphi, 20 * [120], rhofn)  # shapes as above


# Standardize the anm matrix so that for each j and i, anm[j][:, i] has mean 0 and std 1.
anm_pca, means_orig, stds_orig = standardize_anm(anm_Cu)                    # means_orig, stds_orig store the original means and stds for each (j, i) to unstandardize later
how_much_var = 58                                                           # e.g., dim=24 gives 95% of variance, dim=34 gives 99%, and dim=58 gives 99.97%.

n_components = how_much_var                                                 # PCA latent space dimensions
latent_spaces = []                                                          # store latent coordinates (256 × n_components)
pca_models = []                                                             # store PCA models
latent_means = []
latent_covs = []

for k in range(anm_pca.shape[2]):  # iterate over 20 functions
    X = anm_pca[:, :, k]  # shape (256, 180)

    # Center the data (important!)
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Fit PCA across slices
    pca = PCA(n_components=n_components, random_state=SEED)
    Z = pca.fit_transform(X_centered)                                       # shape (256, n_components)

    latent_spaces.append(Z)
    pca_models.append(pca)
    latent_means.append(Z.mean(axis=0))
    latent_covs.append(np.cov(Z, rowvar=False))
    
""" --- Generating Synthetic a_n^m Coefficients by Sampling from Latent Spaces ---"""    
anm_pca_synthetic = np.zeros((n_samples, anm_pca.shape[1], anm_pca.shape[2]))           # (n_samples, 180, 20), because we want to create 'n_samples' number of central slices for our training data

for k in range(anm_pca.shape[2]):  # for each function
    mean_Z = latent_means[k]
    cov_Z = latent_covs[k]
    pca = pca_models[k]

    # Sample new latent coordinates for 256 slices
    Z_new = np.random.multivariate_normal(mean_Z, cov_Z, size=n_samples)                # (256, n_components)

    # Optional: smooth slightly along slices for realistic variation
    # from scipy.ndimage import gaussian_filter1d
    # Z_new_smooth = gaussian_filter1d(Z_new, sigma=1, axis=0)

    # Reconstruct new coefficients
    X_new = pca.inverse_transform(Z_new)
    
    # Add back the original mean (since PCA used centered data)
    X_new += anm_pca[:, :, k].mean(axis=0, keepdims=True)

    anm_pca_synthetic[:, :, k] = X_new

""" --- The new sampled a_n^m coefficients are given the same mean/std as the original copper dataset ---"""
anm_synth, _, _ = standardize_anm(anm_pca_synthetic)
anm_synth = unstandardize_anm(anm_synth, means_orig, stds_orig)

anm_copper_normalized = unstandardize_anm(anm_Cu, means_orig, stds_orig)
copper_getrho_normalized = getrho_anm_synth(order, ncoeffs, rhofn_PCA, anm_Cu, xsize)       # So that when we sample rho_0 from copper, it's consistent with the required central slice maxima


# rhoreturn_synth = getrho_anm_synth(raw_projs, order, pang, nphi, ncoeffs, rhofn_PCA, anm_synth)
rhoreturn_synth = getrho_anm_synth(order, ncoeffs, rhofn_PCA, anm_synth, xsize)
rhoreturn_synth_norm, max_vals_synth = normalize_rhoreturn_ideal_Cu(rhoreturn_synth)




#############################################################################

""" --- Making slight corrections to the synthetic rho_n functions, and substituting the isotropic term with one from Copper ---"""
# List to collect only successful candidates
successful_slices = []


for idx in range(rhoreturn_synth_norm.shape[2]):
    # Smooth all slices for this idx
    smoothed = np.empty_like(rhoreturn_synth_norm[:, :, idx])
    for i in range(rhoreturn_synth_norm.shape[1]):
        smoothed[:, i] = gaussian_filter1d(rhoreturn_synth_norm[:, i, idx], sigma=sigma)

    success = False
    for attempt in range(10):
        rand_val = custom_randint()
        # cu_slice = rhoreturn_ideal_Cu[:, 0, rand_val]
        cu_slice = copper_getrho_normalized[:, 0, rand_val]
        target_max = smoothed[:, 0].max()
        if cu_slice.max() == 0:
            continue

        summed = np.sum(smoothed, axis=1)

        # If rand_val is in 60..116, rescale candidate to the largest possible value so that candidate + summed >= -0.1*target_max everywhere
        if 60 <= rand_val <= 133:
            mask_pos = cu_slice > 0
            mask_neg = cu_slice < 0

            scale_min = -np.inf
            if np.any(mask_pos):
                scale_min = np.max((-0.005 * target_max - summed[mask_pos]) / cu_slice[mask_pos])
            scale_max = np.inf
            if np.any(mask_neg):
                scale_max = np.min((-0.005 * target_max - summed[mask_neg]) / cu_slice[mask_neg])

            orig_scale = target_max / cu_slice.max()
            final_scale = min(orig_scale, scale_max)
            if final_scale < scale_min or final_scale <= 0:
                continue
            candidate = cu_slice * final_scale
        else:
            # Use original scaling
            scale = target_max / cu_slice.max()
            candidate = cu_slice * scale

            # Check the condition
            if np.all(candidate + summed >= -0.1 * target_max):
                smoothed[:, 0] = candidate
                successful_slices.append(smoothed)
                success = True
                break
    # If not successful after 3 attempts, skip this idx

# # Stack only successful slices
# if successful_slices:
#     rhoreturn_ideal_smoothed2 = np.stack(successful_slices, axis=2)
#     print("Created rhoreturn_ideal_smoothed2 with shape:", rhoreturn_ideal_smoothed2.shape)
#     print("Number of successful candidates:", rhoreturn_ideal_smoothed2.shape[2])
# else:
#     print("No successful candidates found.")
    
rhoreturn_ideal_smoothed2 = np.stack(successful_slices, axis=2)
rhoreturn_ideal_smoothed_central = adjust_rho0_maxima(rhoreturn_ideal_smoothed2, rhoreturn_ideal_Cu, percent=0.025)


#############################################################################

""" --- Generating 3D TPMDs using Dynamic Mode Decomposition --- """
Synthetic_Central_Slices = calcplane(rhoreturn_ideal_smoothed2, rhoreturn_ideal_smoothed2.shape[2], order)
Synthetic_Central_Slices_rho0_adjusted = calcplane(rhoreturn_ideal_smoothed_central, rhoreturn_ideal_smoothed_central.shape[2], order)

angles_to_extract = np.linspace(0, 45, 20, endpoint=True)  # 20 values from 0 to 45 inclusive: [0, 2.37, 4.74, ..., 45]
pangzr = angles_to_extract
nprojzr = len(pangzr)                                                       # Number of projections (6 in this case)
ncoeffs = [120] * nprojzr                                                   # Number of coefficients for each projection (Fourier/Zernike expansion)

order = 4                                                                   # Symmetry order (e.g., 4 for C4 symmetry)
nphi = 180                                                                  # Number of angular bins for Fourier expansion (e.g., 45, as long as it's larger than nproj)


""" --- Compute Radon Transform Projections of all the Synthetic TPMD Central Slices --- """
## Synthetic_Central_Slices_rho0_adjusted highlights anisotropic features better, but Synthetic_Central_Slices is more realistic
projections_stacked = compute_projections_stacked(
    Synthetic_Central_Slices,
    # Synthetic_Central_Slices_rho0_adjusted,
    angles_to_extract
)

pangzr = angles_to_extract
nprojzr = len(pangzr)                                                       # Number of projections (6 in this case)
ncoeffs = [120] * nprojzr                                                   # Number of coefficients for each projection (Fourier/Zernike expansion)

order = 4                                                                   # Symmetry order (e.g., 4 for C4 symmetry)
nphi = 180

""" --- Recover the a_n^m Chebyshev components for the Synthetic TPMD Central Slices --- """
_, anm_arr = getrho_training_data(projections_stacked, order, pangzr, nphi, ncoeffs, rhofn_PCA)

print(anm_arr.shape)

model = train_pca_dmd_per_channel(anm_Cu, latent_dim=24)


centre = xsize
all_indices = np.arange(centre, centre + 102)
num_samples = 6
n_items = anm_arr.shape[0]
y_indices_list = precompute_y_indices(all_indices, num_samples, centre, n_items, rng)


# for i in tqdm(range(1), desc="Generating Projections, Rhos, and TPMD datasets"):
for i in tqdm(range(anm_arr.shape[0]), desc="Generating Projections, Rhos, and TPMD datasets"):
    predicted = rollout_per_channel(model, anm_arr[i], n_steps=256)
    rhos_evolved = anm_to_rhos(predicted, order, rhofn, ncoeffs)

    # print(predicted.shape)
    # print(rhos_evolved.shape)

    Rand_vol_evolved = calcplane(rhos_evolved, rhos_evolved.shape[2], order)

    """ --- Taking Ideal Projections from the DMD 3D TPMD --- """
    synth_tpmd_projections = compute_projections_stacked(
        Rand_vol_evolved,
        angles_to_extract
    )
    full_synth_tpmd_projections = np.concatenate([np.flip(synth_tpmd_projections, axis=1), synth_tpmd_projections], axis=1)

    """ --- Simulating Realistic Measurement Projections from the Ideal Projections --- """
    # Apply experimental noise and realistic conditions to the projections
    comp_pang = np.array([0, 12.2, 24, 35, 45])
    # comp_pang = np.linspace(0, 45, num=5)
    selected_indices = [np.abs(pang - val).argmin() for val in comp_pang]
    pang_measure = pang[selected_indices]
    measure_projs = upsample(np.array([full_synth_tpmd_projections[:, :, i] for i in selected_indices]).transpose(1, 2, 0))
    

    # reconstruction, _ = getrho(full_synth_tpmd_projections, order, pang, nphi, 20 * [120], rhofn_PCA)
    # MCM = calcplane(reconstruction, reconstruction.shape[2], order)
    # reconstruction_measure, _ = getrho(realistic_projs, order, pang_measure, nphi, 5 * [120], rhofn_PCA)
    # MCM_measure = calcplane(reconstruction_measure, reconstruction_measure.shape[2], order)
    
    
    proj_ideal_dir = os.path.join(base_dir, "Proj_Ideal")
    os.makedirs(proj_ideal_dir, exist_ok=True)

    rho_ideal_dir = os.path.join(base_dir, "Rho_Ideal")
    os.makedirs(rho_ideal_dir, exist_ok=True)

    tpmd_ideal_dir = os.path.join(base_dir, "TPMD_Ideal")
    os.makedirs(tpmd_ideal_dir, exist_ok=True)
    
    # # Indices to extract (rounded to nearest integer)
    # centre = full_synth_tpmd_projections.shape[1] // 2
    # increments = 10
    # all_indices = np.arange(centre, centre + 102)  # inclusive of centre+101
    # num_samples = 6  # total number of indices you want (including centre)
    # sampled = rng.choice(all_indices[1:], size=num_samples - 1, replace=False)  # exclude centre for sampling
    # y_indices = np.concatenate(([centre], sampled))
    # y_indices = np.sort(y_indices)
    # # y_indices = np.round(np.arange(centre, centre + 50, increments)).astype(int)       # Saving projection from centre (idx 256) to
    
    y_indices = y_indices_list[i]

    """ --- Saving the Projection Data --- """
    save_ideal_proj = full_synth_tpmd_projections[:, y_indices, :]
    

    # Should have shape (N, len(y_indices), nproj) ––> flattened to 512 * 5 * 20 = 51,200
    save_path_ideal = os.path.join(proj_ideal_dir, f"proj_ideal_{i}.txt")
    np.savetxt(save_path_ideal, save_ideal_proj.flatten())
    
    
    """ --- Saving the 3D Rho Data --- """
    reconstruction_ideal, _ = getrho_training_data(save_ideal_proj, order, pang, nphi, 20 * [120], rhofn_PCA)
    # print(reconstruction_ideal.shape)
    
    # Should have shape (N/2, nproj, len(y_indices)) ––> flattened to 256 * 20 * 5 = 25,600
    save_path_ideal = os.path.join(rho_ideal_dir, f"rho_ideal_{i}.txt")
    np.savetxt(save_path_ideal, reconstruction_ideal.flatten())
    
    """ --- Saving the 3D TPMD Data (Upper-Left Quadrant) --- """
    MCM_ideal = calcplane(reconstruction_ideal, reconstruction_ideal.shape[2], order)
    
    
    # Should have shape (N/2, len(y_indices), N/2) ––> flattened to 256 * 5 * 256 = 327,680
    save_path_ideal = os.path.join(tpmd_ideal_dir, f"tpmd_ideal_{i}.txt")
    np.savetxt(save_path_ideal, MCM_ideal[:centre, :, :centre].flatten())
    
    
    for counts in count_ttl:
        
        base_dir_meas = os.path.join(base_dir, f"{counts // 1_000_000}M_Counts")

        proj_meas_dir = os.path.join(base_dir_meas, "Proj_Measurement")
        os.makedirs(proj_meas_dir, exist_ok=True)

        rho_meas_dir = os.path.join(base_dir_meas, "Rho_Measurement")
        os.makedirs(rho_meas_dir, exist_ok=True)

        tpmd_meas_dir = os.path.join(base_dir_meas, "TPMD_Measurement")
        os.makedirs(tpmd_meas_dir, exist_ok=True)
                

        # Apply realistic projection processing
        realistic_projs = make_realistic_projections(
            measure_projs,
            sigma_x=0.11, sigma_y=0.137,
            projection_size_au=5.0,
            total_counts=counts,
            four_sym=True
        )
        
        """ --- Saving the Projection Data --- """
        save_meas_proj = realistic_projs[:, y_indices, :]
    
        # Should have shape (N, len(y_indices), nproj) ––> flattened to 512 * 5 * 5 = 12,800
        save_path_meas = os.path.join(proj_meas_dir, f"proj_meas_{i}.txt")
        np.savetxt(save_path_meas, save_meas_proj.flatten())
    
    
        """ --- Saving the 3D Rho Data --- """
        reconstruction_measure, _ = getrho_training_data(save_meas_proj, order, pang_measure, nphi, 5 * [120], rhofn_PCA)

        # Should have shape (N/2, nproj, len(y_indices)) ––> flattened to 256 * 5 * 5 = 6,400
        save_path_meas = os.path.join(rho_meas_dir, f"rho_meas_{i}.txt")
        np.savetxt(save_path_meas, reconstruction_measure.flatten())
    
    
        """ --- Saving the 3D TPMD Data (Upper-Left Quadrant) --- """
        MCM_measure = calcplane(reconstruction_measure, reconstruction_measure.shape[2], order)
    
        # Should have shape (N/2, len(y_indices), N/2) ––> flattened to 256 * 5 * 256 = 327,680
        save_path_meas = os.path.join(tpmd_meas_dir, f"tpmd_meas_{i}.txt")
        np.savetxt(save_path_meas, MCM_measure[:centre, :, :centre].flatten())


        """ --- Visualizing and Saving Side-by-Side Images of Ideal vs Measurement TPMD Slices --- """
        # Visualize and save the 3D TPMD Data as side-by-side images
        for j in range(MCM_ideal.shape[1]):
            vmin = min(np.quantile(MCM_measure[:, j, :], 0.00), np.quantile(MCM_ideal[:, j, :], 0.00))
            vmax = max(np.quantile(MCM_measure[:, j, :], 1.00), np.quantile(MCM_ideal[:, j, :], 1.00))

            fig, axes = plt.subplots(1, 3, figsize=(12, 6))
            im0 = axes[0].imshow(MCM_ideal[:, j, :], cmap='hot', aspect='equal', origin='lower', vmin=vmin, vmax=vmax)
            axes[0].axis('off')
            # axes[0].set_title('Ideal')
            # plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, location="left")
            
            im1 = axes[1].imshow(MCM_measure[:, j, :], cmap='hot', aspect='equal', origin='lower', vmin=vmin, vmax=vmax)
            axes[1].axis('off')
            # axes[1].set_title('Measurement')
            # plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            percent_diff = 100 * (MCM_ideal[:, j, :] - MCM_measure[:, j, :]) / (np.max(np.abs(MCM_measure[:, j, :])))
            perc_range = np.max(np.abs(percent_diff))

            im2 = axes[2].imshow(np.abs(percent_diff), cmap='seismic', origin='lower', aspect='equal', vmin=-perc_range, vmax=perc_range)
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01)
            
            tpmd_folder = os.path.join(base_dir_meas, "TPMD_Image", f"TPMD_{i}")
            os.makedirs(tpmd_folder, exist_ok=True)
            save_img_path = os.path.join(tpmd_folder, f"Slice_{y_indices[j] - 256}.png")
            plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0)
            
            plt.close(fig)


print("Data generation complete.")