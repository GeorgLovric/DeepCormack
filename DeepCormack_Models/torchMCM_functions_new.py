import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import os
import re

def plnorm(project: torch.Tensor) -> torch.Tensor:
    """
    Normalize all projections to the total sum of the first projection (column 0).
    Only columns 1..nproj-1 are rescaled; column 0 is unchanged.

    Args:
        project (torch.Tensor): Input tensor of shape (xsize, nproj).

    Returns:
        torch.Tensor: Normalized projections of shape (xsize, nproj).
    """
    sum1 = torch.sum(project[:, 0])
    nproj = project.shape[1]
    for n in range(nproj):
        sump = torch.sum(project[:, n])
        if sump != 0:
            project[:, n] = project[:, n] * (sum1 / sump)
        else:
            project[:, n] = 0
    return project

def calccos(project: torch.Tensor, xsize: int, nphi: int, nproj: int) -> torch.Tensor:
    """
    Evaluate data as a function of cos(phi) using PyTorch.

    Args:
        project (torch.Tensor): Input tensor of shape (xsize, nproj).
        xsize (int): Number of radial bins.
        nphi (int): Number of angular bins.
        nproj (int): Number of projections.

    Returns:
        torch.Tensor: Interpolated projections of shape (nphi, nproj).
    """
    angles = torch.linspace(0.0, 90.0, nphi, device=project.device) * torch.pi / 180.0
    dists = (xsize - 1) * torch.cos(angles)
    i = torch.floor(dists).long()
    deltax = dists - i
    valid = (i >= 0) & (i < xsize - 1)
    proj = torch.zeros((nphi, nproj), dtype=project.dtype, device=project.device)
    proj[valid] = (project[i[valid], :] * (1 - deltax[valid]).unsqueeze(1) +
                   project[i[valid] + 1, :] * deltax[valid].unsqueeze(1))
    proj[~valid] = project[xsize - 1, :].unsqueeze(0)
    return proj

def setupsin(nphi: int) -> torch.Tensor:
    """
    Build and invert a sin matrix using PyTorch.

    Args:
        nphi (int): Number of angular bins.

    Returns:
        torch.Tensor: Flattened (Fortran order) inverse sine matrix.
    """
    step = 90.0 / nphi
    conv = torch.pi / 180.0
    fa = ((torch.arange(nphi) + 1) * step * conv).reshape(-1, 1)
    j_idx = (2 * (torch.arange(nphi) + 1) - 1).reshape(1, -1)
    sinmat = torch.sin(j_idx * fa)
    sininv = torch.linalg.inv(sinmat)
    return sininv.t().reshape(-1)  # Fortran order flattening

def setupprojs(order: int, pang, nproj: int) -> torch.Tensor:
    """
    Construct and invert the projection matrix using PyTorch.

    Args:
        order (int): Symmetry order for system.
        pang (list or torch.Tensor): Projection angles in degrees.
        nproj (int): Number of projections.

    Returns:
        torch.Tensor: Inverted projection matrix of shape (nproj, nproj).
    """
    conv = torch.pi / 180.0
    pang = torch.tensor(pang, dtype=torch.float32)
    angles = order * conv * pang
    j_idx = torch.arange(nproj, dtype=angles.dtype)
    simul = torch.cos(torch.outer(angles, j_idx))
    simul_inv = torch.linalg.inv(simul)
    return simul_inv

def calcanm(sinmat: torch.Tensor, simul: torch.Tensor, nphi: int, proj: torch.Tensor) -> torch.Tensor:
    """
    Vectorized calculation of anm using PyTorch.

    Args:
        sinmat (torch.Tensor): Flattened (Fortran order) inverse sine matrix (length nphi*nphi).
        simul (torch.Tensor): Inverted projection matrix (nproj, nproj).
        nphi (int): Number of angular bins.
        proj (torch.Tensor): Interpolated projections (nphi, nproj).

    Returns:
        torch.Tensor: anm coefficients (nphi, nproj).
    """
    sininv = sinmat.reshape(nphi, nphi).t()  # Fortran order
    projf = proj @ simul.t()
    anm = 0.5 * (sininv @ projf)
    return anm

def zernike_recursion(r: torch.Tensor, nn: int, coeff_count: int) -> torch.Tensor:
    """
    Compute Zernike-like polynomials for all r (vectorized over r).

    Args:
        r (torch.Tensor): Radial grid (1D).
        nn (int): Order parameter.
        coeff_count (int): Number of coefficients.

    Returns:
        torch.Tensor: Zernike polynomials (len(r), coeff_count).
    """
    zern = torch.zeros((len(r), coeff_count), dtype=r.dtype, device=r.device)
    if coeff_count == 0:
        return zern
    zern[:, 0] = 1.0 if nn == 0 else r ** nn
    if coeff_count > 1:
        zern[:, 1] = zern[:, 0] * ((nn + 2) * r ** 2 - (nn + 1))
    for l in range(2, coeff_count):
        m = l - 1
        m2 = nn + 2 * m
        m1 = m2 + 2
        num = ((nn + l + m) * (m2 * (m1 * r ** 2 - nn - 1) - 2 * m ** 2) * zern[:, l - 1] -
               m * (nn + m) * m1 * zern[:, l - 2])
        denom = l * m2 * (nn + l)
        zern[:, l] = num / denom
    return zern

def precompute_zernike(xsize: int, nproj: int, order: int, nphi: int, ncoeff) -> tuple:
    """
    Precompute radial grid, angle step and per-projection coeff_count.

    Args:
        xsize (int): Number of radial bins.
        nproj (int): Number of projections.
        order (int): Symmetry order.
        nphi (int): Number of angular bins.
        ncoeff (int or list): Number of coefficients.

    Returns:
        tuple: (r, coeff_count, nn, ni)
    """
    delrho = 1.0 / (xsize - 1)
    r = torch.arange(xsize, dtype=torch.float32) * delrho
    deltaphi = 90.0 / nphi
    conv = torch.pi / 180.0
    deltaphi_rad = deltaphi * conv
    nn = torch.arange(nproj, dtype=torch.int64) * order
    ni = nn // 2

    if isinstance(ncoeff, int):
        ncoeff_arr = torch.full((nproj,), ncoeff, dtype=torch.int64)
    else:
        ncoeff_arr = torch.tensor(ncoeff, dtype=torch.int64)
        if ncoeff_arr.numel() != nproj:
            ncoeff_arr = ncoeff_arr.repeat(nproj)[:nproj]

    avail = (nphi - ni).clamp(min=1)
    m_max = torch.floor(0.5 * (torch.pi / deltaphi_rad - (nn + 1))).to(torch.int64)
    m_max = torch.maximum(m_max, torch.ones_like(m_max))
    coeff_count = torch.minimum(torch.minimum(ncoeff_arr, avail), m_max)
    return r, coeff_count, nn, ni

def calcrho(anm: torch.Tensor, order: int, nproj: int, xsize: int, nphi: int, rhofn: torch.Tensor, ncoeff, consistency_condition: bool = True) -> torch.Tensor:
    """
    Calculate the radial density functions, rho_{n}.

    Args:
        anm (torch.Tensor): anm coefficients (nphi, nproj).
        order (int): Symmetry order.
        nproj (int): Number of projections.
        xsize (int): Number of radial bins.
        nphi (int): Number of angular bins.
        rhofn (torch.Tensor): Fermi cutoff function (xsize,).
        ncoeff (int or list): Number of coefficients.
        consistency_condition (bool): Whether to zero small anm parts if nn>0.

    Returns:
        torch.Tensor: rho (xsize, nproj).
    """
    rho = torch.zeros((xsize, nproj), dtype=anm.dtype, device=anm.device)
    r, coeff_count, nn, ni = precompute_zernike(xsize, nproj, order, nphi, ncoeff)
    for ii in range(nproj):
        anm_slice = anm[ni[ii]:nphi, ii].clone()
        if consistency_condition and nn[ii] > 0:
            anm_slice[:nn[ii]//2] = 0
        if coeff_count[ii] == 0:
            continue
        zern = zernike_recursion(r, int(nn[ii]), int(coeff_count[ii]))
        weights = (2 * (torch.arange(coeff_count[ii], device=anm.device) + 1) - 1 + nn[ii]) * anm_slice[:coeff_count[ii]]
        contrib = zern @ weights
        rho[:, ii] = contrib * rhofn
    return rho

def rhocutoff(xsize: int, rhocut: int, flvl: float, kt: float) -> torch.Tensor:
    """
    Generate Fermi cutoff function.

    Args:
        xsize (int): Number of radial bins.
        rhocut (int): If 0, apply cutoff; else, return ones.
        flvl (float): Center of cutoff.
        kt (float): Slope.

    Returns:
        torch.Tensor: Fermi cutoff (xsize,).
    """
    if rhocut == 0:
        x = torch.arange(xsize, dtype=torch.float32) - flvl
        f = 1.0 / (1.0 + torch.exp(x / kt))
        f[f < 1e-7] = 0.0
        return f
    else:
        return torch.ones(xsize, dtype=torch.float32)




def getrho(rawdat, order, pang, nphi, ncoeff, rhofn):
    """
    Compute rho using PyTorch tensors (GPU compatible).

    Args:
        rawdat (torch.Tensor): Input data of shape (nsize, nsize, nproj).
        order (int): Symmetry order.
        pang (list or torch.Tensor): Projection angles in degrees.
        nphi (int): Number of angular bins.
        ncoeff (int or list): Number of coefficients.
        rhofn (tuple): (rhocut, flvl, kt) for Fermi cutoff.

    Returns:
        torch.Tensor: rho of shape (xsize, nproj, xsize).
        torch.Tensor: anm matrix of shape (xsize, nphi, nproj).
    """
    device = rawdat.device
    nproj = rawdat.shape[2]
    nsize = rawdat.shape[0]
    xsize = nsize // 2
    xstart = nsize // 2

    simul = setupprojs(order, pang, nproj).to(device)
    sinmat_flat = setupsin(nphi).to(device)
    sininv_mat = sinmat_flat.reshape(nphi, nphi).t()  # Fortran order

    rhocut, flvl, kt = rhofn
    rhofn_array = rhocutoff(xsize, rhocut, flvl, kt).to(device)

    deltaphi = 90.0 / nphi
    angles = torch.linspace(0.0, 90.0, nphi, device=device) * torch.pi / 180.0
    dists = (xsize - 1) * torch.cos(angles)
    i_idx = torch.floor(dists).long()
    deltax = dists - i_idx
    valid_mask = (i_idx >= 0) & (i_idx < xsize - 1)

    r, coeff_count, nn, ni = precompute_zernike(xsize, nproj, order, nphi, ncoeff)
    r = r.to(device)
    coeff_count = coeff_count.to(device)
    nn = nn.to(device)
    ni = ni.to(device)

    rhoreturn = torch.zeros((xsize, nproj, xsize), dtype=rawdat.dtype, device=device)
    anm_matrix = torch.zeros((xsize, nphi, nproj), dtype=rawdat.dtype, device=device)

    for yfixed in range(xsize):
        yfixed_shifted = yfixed + xstart
        project = rawdat[xstart:xstart + xsize, yfixed_shifted, :].clone()
        project = plnorm(project)
        proj = calccos(project, xsize, nphi, nproj)
        anm = calcanm(sinmat_flat, simul, nphi, proj)
        anm_matrix[yfixed] = anm

        rho = calcrho(anm, order, nproj, xsize, nphi, rhofn_array, coeff_count)
        rhoreturn[:, :, yfixed] = rho

    return rhoreturn, anm_matrix

def calcplane(rhos):
    """
    Reconstructs a stack of 2D planes from radial densities using PyTorch (GPU compatible).

    Args:
        rhos (torch.Tensor): Input tensor of shape (xsize, nproj, n_slices).
        order (int): Symmetry order.

    Returns:
        torch.Tensor: Stack of reconstructed planes (2*xsize, 2*xsize).
    """
    device = rhos.device
    xsize = rhos.shape[0]
    nproj = rhos.shape[1]
    order = 4
    N_full = 2 * xsize

    x = torch.arange(N_full, device=device)
    z = torch.arange(N_full, device=device)
    x_grid, z_grid = torch.meshgrid(x, z, indexing='ij')
    x0 = xsize - 0.5
    zdist = torch.abs(z_grid - x0).clamp(min=1e-3)
    theta = torch.atan2(x_grid - x0, zdist)
    p = torch.sqrt((torch.abs(x_grid - x0)) ** 2 + zdist ** 2)
    intp = torch.floor(p).long()
    deltap = p - intp
    idx1 = intp.clamp(0, xsize - 2)
    idx2 = (intp + 1).clamp(0, xsize - 1)
    mask_exact_top = (intp == (xsize - 1))

    n_idx = torch.arange(nproj, device=device).view(1, 1, -1)
    cos_terms = torch.cos(n_idx * order * theta.unsqueeze(-1))

    plane_stack = torch.zeros((N_full, N_full), dtype=rhos.dtype, device=device)

    rhos_y = rhos[:, :nproj]  # (xsize, nproj)
    r1 = torch.stack([rhos_y[idx1, n] for n in range(nproj)], dim=-1)
    r2 = torch.stack([rhos_y[idx2, n] for n in range(nproj)], dim=-1)
    interp = (1.0 - deltap).unsqueeze(-1) * r1 + deltap.unsqueeze(-1) * r2

    if mask_exact_top.any():
        last_row = rhos_y[xsize - 1, :].view(1, 1, nproj)
        interp[mask_exact_top] = last_row

    plane = torch.sum(interp * cos_terms, dim=2)
    plane = torch.clamp(plane, min=0.0)
    plane_stack[:, :] = plane

    return plane_stack




from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import rcParams

n_levels = 16
colors = [
    "#000002", "#000557", "#0012a7", "#0820f8",
    "#520da6", "#550056", "#a80100", "#fc0300",
    "#ff5804", "#fe9947", "#fca305", "#aaa400",
    "#fefd35", "#fefd58", "#fefe98", "#fefdd2",
]
rcParams.update(
    {
        'font.size': 18,
        # Times New Roman needs to be installed
        'font.family': 'serif',
        'font.serif': 'DejaVu Serif',
        'figure.figsize': (12, 8),
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'legend.fontsize': 12,
    }
)
cmap = ListedColormap(colors)

def plt_TPMD(recon_image):
    fig, ax = plt.subplots()
    bounds = np.linspace(recon_image.min().item(), recon_image.max().item(), n_levels + 1)
    norm = BoundaryNorm(bounds, ncolors=n_levels)
    img = ax.imshow(recon_image.cpu().numpy(), cmap=cmap, norm=norm)
    
    cbar = fig.colorbar(
        img, ax=ax, boundaries=bounds, ticks=[bounds[0], bounds[-1]], aspect=30, pad=0.01
    )
    #cbar.set_label("Reconstructed Intensity")
    cbar.ax.set_yticklabels(['min', 'max'])
    
    title = ax.set_title(f"Reconstructed Central Slice")
    ax.set_xlabel("Px")
    ax.set_ylabel("Pz")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()

def plot_and_compare_reconstructions(
    targ_np: np.ndarray,
    meas_np: np.ndarray,
    output_np: np.ndarray,
    count_rate: int = None,
    save_cu: bool = False,
    save_dir: str = "/store/LION/gfbl2/Training_Data/Figures/Cu",
    cmap=cmap
):
    """
    Plot and compare ground truth, Cormack input, and UNet output reconstructions,
    show percentage errors, print MSE/SSIM, and optionally save the figure.

    Parameters
    ----------
    targ_np : np.ndarray
        Ground truth (target) image.
    meas_np : np.ndarray
        Cormack (input) image.
    output_np : np.ndarray
        UNet (output) image.
    count_rate : int, optional
        Count rate for filename (used if save_cu is True).
    save_cu : bool
        If True, saves the figure to save_dir.
    save_dir : str
        Directory to save the figure if save_cu is True.
    cmap : str or Colormap, optional
        Colormap to use for images (default 'hot').
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from skimage.metrics import structural_similarity as ssim
    import numpy as np
    import os

    plt.rcParams['font.family'] = plt.rcParamsDefault['font.family']

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], hspace=0.1, wspace=0.05)

    # First row
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(targ_np, cmap=cmap)
    ax0.set_title('Ground Truth (Target)', fontsize=18, fontweight='bold')
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1.imshow(meas_np, cmap=cmap)
    ax1.set_title('Cormack (Input)', fontsize=18, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(output_np, cmap=cmap)
    ax2.set_title('MLP (Output)', fontsize=18, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Second row: skip [1, 0]
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    residual_cormack_np = targ_np - meas_np
    residual_unet_np = targ_np - output_np

    denominator_cormack = np.max(np.abs(meas_np))
    cormack_perc_diff = residual_cormack_np / (denominator_cormack if denominator_cormack != 0 else 1) * 100

    denominator_unet = np.max(np.abs(output_np))
    unet_perc_diff = residual_unet_np / (denominator_unet if denominator_unet != 0 else 1) * 100

    perc_max = max(np.max(np.abs(cormack_perc_diff)), np.max(np.abs(unet_perc_diff)))
    colormap = 'bwr'
    im3 = ax3.imshow(cormack_perc_diff, cmap=colormap, vmin=-perc_max, vmax=perc_max)
    im4 = ax4.imshow(unet_perc_diff, cmap=colormap, vmin=-perc_max, vmax=perc_max)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Shared colorbar
    cbar_ax = fig.add_axes([0.883, 0.11, 0.01, 0.367])
    cb = fig.colorbar(im4, cax=cbar_ax, orientation='vertical')
    cb.ax.tick_params(labelsize=12)

    # Add vertical row titles
    fig.text(0.12, 0.70, 'Reconstruction', va='center', ha='center', rotation='vertical', fontsize=22, fontweight='bold')
    fig.text(0.12, 0.3, 'Percentage Error', va='center', ha='center', rotation='vertical', fontsize=22, fontweight='bold')

    # Compute MSE & SSIM
    unet_mse = np.mean((targ_np - output_np) ** 2)
    cormack_mse = np.mean((targ_np - meas_np) ** 2)
    unet_ssim = ssim(targ_np, output_np, data_range=targ_np.max() - targ_np.min())
    cormack_ssim = ssim(targ_np, meas_np, data_range=targ_np.max() - targ_np.min())

    print(f"{'':10s} {'Cormack':>12s} {'UNet':>12s}")
    print(f"{'MSE':10s} {cormack_mse:12.3e} {unet_mse:12.3e}")
    print(f"{'SSIM':10s} {cormack_ssim:12.4f} {unet_ssim:12.4f}\n")

    mse_better = unet_mse < cormack_mse
    ssim_better = unet_ssim > cormack_ssim

    if mse_better and ssim_better:
        print("MLP SSIM and MSE Better")
    elif mse_better:
        print("MLP MSE Better")
    elif ssim_better:
        print("MLP SSIM Better")
    else:
        print("MLP is Worse")
    print()

    if save_cu:
        os.makedirs(save_dir, exist_ok=True)
        if count_rate is not None:
            save_path = os.path.join(save_dir, f"Cu_UNet_Prediction{count_rate//1_000_000}M.png")
        else:
            save_path = os.path.join(save_dir, "Cu_UNet_Prediction.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

def load_randrhos(folder, xsize, nproj, measurement="rho_rand_measurement", max_samples=None, dtype=torch.float64, device='cpu'):
    """
    Loads and reshapes simulated rho files, returning a PyTorch tensor.
    For measurement: loads all rho_meas_{idx}.txt files and returns shape (num_files, xsize, nproj, 5).
    For ideal: loads all rho_ideal_{idx}.txt files and returns shape (num_files, xsize, nproj, 5).
    """
    if measurement == "rho_rand_ideal":
        pattern = r"rho_ideal_(\d+)\.txt"
    elif measurement == "rho_rand_measurement":
        pattern = r"rho_meas_(\d+)\.txt"
    else:
        raise ValueError("Invalid measurement type")

    files = [f for f in os.listdir(folder) if re.match(pattern, f)]
    indices = []
    for f in files:
        match = re.search(r"(\d+)", f)
        if match:
            indices.append(int(match.group(1)))
        else:
            indices.append(-1)  # fallback if no match

    sorted_files = [f for _, f in sorted(zip(indices, files)) if _ != -1]
    if max_samples is not None:
        sorted_files = sorted_files[:max_samples]
    rhos = []
    for fname in sorted_files:
        arr = np.loadtxt(os.path.join(folder, fname)).reshape((xsize, nproj, 5))
        rhos.append(arr)
    arr_np = np.stack(rhos, axis=0)
    return torch.tensor(arr_np, dtype=dtype, device=device)
        

# ============================================================
# Symmetry utilities
# ============================================================

def expand_quadrant_to_full(quadrant):
    """
    quadrant: (B, 5, 256, 256)
    returns : (B, 5, 512, 512)
    """
    top = torch.cat([quadrant, torch.flip(quadrant, dims=[3])], dim=3)
    bottom = torch.cat(
        [torch.flip(quadrant, dims=[2]),
         torch.flip(torch.flip(quadrant, dims=[2]), dims=[3])],
        dim=3
    )
    return torch.cat([top, bottom], dim=2)


# ============================================================
# Image reconstruction
# ============================================================

def reconstruct_image_batch(rhos):
    """
    Vectorized batch reconstruction of images from RDFs.
    Args:
        rhos: (B, xsize, nproj, nch)
    Returns:
        imgs: (B, nch, 2*xsize, 2*xsize)
    """
    # rhos: (B, xsize, nproj, nch)
    B, xsize, nproj, nch = rhos.shape
    device = rhos.device
    # Output: (B, nch, 2*xsize, 2*xsize)
    imgs = []
    for c in range(nch):
        # For each channel, process the whole batch at once
        # rhos[:, :, :, c]: (B, xsize, nproj)
        # calcplane expects (xsize, nproj, n_slices) so we transpose
        # We'll stack along a new axis for batch
        # Vectorize: move batch to last axis, then process all at once
        # (B, xsize, nproj) -> (xsize, nproj, B)
        rhos_c = rhos[:, :, :, c].permute(1,2,0)  # (xsize, nproj, B)
        # Vectorized calcplane for all B (see below for details)
        # We'll need to update calcplane to support batch dim if not already
        # For now, process each batch in a list comprehension (still faster than nested for-loops)
        planes = torch.stack([calcplane(rhos_c[:,:,b]) for b in range(B)], dim=0)  # (B, 2*xsize, 2*xsize)
        imgs.append(planes)
    imgs = torch.stack(imgs, dim=1)  # (B, nch, 2*xsize, 2*xsize)
    return imgs



def compute_tpmd(input_tensor):
    """
    Compute the TPMD for a batch of input tensors by concatenating the result for each theta.

    Args:
        input_tensor (torch.Tensor): shape (batch_size, 256, 5)

    Returns:
        torch.Tensor: shape (batch_size, 256*256)
            The concatenated TPMD for each sample in the batch.
    """
    batch_size, length, n_coeffs = input_tensor.shape
    device = input_tensor.device

    # Compute theta values for each i (0 to 255)
    thetas = torch.arange(length, device=device)  # (256,)
    theta_vals = torch.arctan(thetas / 256.0)     # (256,)

    # Precompute the coefficient multipliers (0, 4, 8, 12, 16)
    coeff_multipliers = torch.arange(n_coeffs, device=device) * 4  # (5,)

    # For each i (0 to 255), compute the sum over n_coeffs for all batch and all positions
    # We'll build a list of (batch_size, 256) tensors, one for each i, then concatenate along dim=1

    tpmd_list = []
    for i in range(length):
        # For each i, compute cos(coeff_multipliers * theta_vals[i])
        cos_terms = torch.cos(coeff_multipliers * theta_vals[i])  # (5,)
        # Multiply input_tensor[:, :, :] by cos_terms, sum over n_coeffs
        # input_tensor[:, j, :] * cos_terms for all j
        # This is a batch matrix multiplication
        tpmd_i = (input_tensor * cos_terms)  # (batch_size, 256, 5)
        tpmd_i_sum = tpmd_i.sum(dim=2)       # (batch_size, 256)
        # For each i, we want the vector for all positions (j=0..255)
        # But the sum is over n_coeffs, so we keep (batch_size, 256)
        # For concatenation, we want to flatten along the 256 axis
        tpmd_list.append(tpmd_i_sum)
    # Concatenate along the 1st axis (dim=1)
    tpmd_concat = torch.cat([x for x in tpmd_list], dim=1)  # (batch_size, 256*256)
    #return tpmd_concat
    # Batch-wise normalization
    max_vals = tpmd_concat.abs().amax(dim=1, keepdim=True)  # (batch_size, 1)
    tpmd_normalized = tpmd_concat / (max_vals + 1e-8)
    return tpmd_normalized


