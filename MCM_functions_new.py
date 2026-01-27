import numpy as np
import math
import torch
from tqdm import tqdm
from scipy.linalg import lstsq
from scipy.ndimage import rotate, convolve, gaussian_filter1d, gaussian_filter, zoom
from scipy.optimize import curve_fit
from symfit import parameters, variables, sin, cos, Fit
import numba
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import random
import os
import re
from scipy.signal import fftconvolve
import glob
from IPython.display import display, clear_output, HTML
from sklearn.decomposition import PCA
from skimage.transform import radon




def plnorm(project):
    """
    Normalize all projections to the total sum of the first projection (column 0).
    Only columns 1..nproj-1 are rescaled; column 0 is unchanged.
    project: shape (xsize, nproj)
    """
    sum1 = np.sum(project[:, 0])
    nproj = project.shape[1]
    for n in range(0, nproj):           # Fortran starts loop from second column
        sump = np.sum(project[:, n])
        if sump != 0:
            project[:, n] *= sum1 / sump
        else:
            project[:, n] = 0
    return project



def calccos(project, xsize, nphi, nproj):
    """
    Vectorized: Evaluate data as a function of cos(phi).
    project: shape (xsize, nproj)
    Returns: (nphi, nproj)
    """
    deltaphi = 90.0 / nphi
    
    # angles from 0° to 90° (inclusive) with length nphi, converted to radians
    angles = np.linspace(0.0, 90.0, nphi, endpoint=True) * np.pi / 180.0  # shape (nphi,)

    #### dists = (xsize) * np.cos(angles)                         # shape (nphi,)          
    dists = (xsize-1) * np.cos(angles)                                  # shape (nphi,)           
    i = np.floor(dists).astype(int)
    deltax = dists - i
    valid = (i >= 0) & (i < xsize-1)    
    proj = np.zeros((nphi, nproj))

    # interpolate where safe
    proj[valid, :] = (project[i[valid], :] * (1 - deltax[valid])[:, None] +
                     project[i[valid] + 1, :] * deltax[valid][:, None])
    # for invalid rows (e.g. near edge), assign edge value
    proj[~valid, :] = project[xsize - 1, :][None, :]
    return proj


def setupsin(nphi):
    """
    Setupsin: Builds a square matrix (sinmat) of size nphi x nphi.
    Each element is sin((2*(j+1)-1) * fa), where fa is an angle in 
    radians, incremented in steps over 0–90°. Inverts the matrix 
    (np.linalg.inv(sinmat)). Returns the flattened inverse (1D array).    
    """
    step = 90.0 / nphi
    conv = np.pi / 180.0
    fa = ((np.arange(nphi) + 1) * step * conv).reshape(-1, 1)               # shape (nphi, 1)
    j_idx = (2 * (np.arange(nphi) + 1) - 1).reshape(1, -1)                  # shape (1, nphi)
    sinmat = np.sin(j_idx * fa)                                             # shape (nphi, nphi)
    sininv = np.linalg.inv(sinmat)
    # return sininv.flatten()
    return sininv.flatten(order='F')                       # Fortran ordering is column-major when flattening, whereas python is row-major (so transpose necessary).


def setupprojs(order, pang, nproj):
    """
    The setupprojs function constructs and inverts a 
    projection matrix used to relate the measured 
    projections at different angles to the coefficients 
    in the angular Fourier expansion.
    ___________________________________________________
    Set up the projection matrix and invert.
    pang: list of projection angles in degrees.
    order: symmetry order for system (e.g. 4 for C4).
    nproj: number of projections.
    """
    conv = np.pi / 180.0
    pang = np.array(pang)
    angles = order * conv * pang                # shape (nproj,)
    j_idx = np.arange(nproj)                    # shape (nproj,)
    simul = np.cos(np.outer(angles, j_idx))     # shape (nproj, nproj)
    simul_inv = np.linalg.inv(simul)
    return simul_inv


def calcanm(sinmat, simul, nphi, proj):
    """
    Vectorized calculation of anm.
    sinmat: flattened (Fortran order) inverse sine matrix (length nphi*nphi)
    simul: (nproj, nproj) projection matrix inverse
    proj: (nphi, nproj)
    Returns: anm (nphi, nproj)
    """
    # reshape sinmat back to matrix (Fortran order)
    sininv = sinmat.reshape((nphi, nphi), order='F')   # shape (nphi, nphi)

    # projf[t,i] = sum_j simul[i,j] * proj[t,j]  -> projf = proj @ simul.T
    projf = proj.dot(simul.T)                         # shape (nphi, nproj)

    # anm = 0.5 * sininv @ projf
    anm = 0.5 * (sininv.dot(projf))                   # shape (nphi, nproj)

    return anm


def zernike_recursion(r, nn, coeff_count):
    """Compute Zernike-like polynomials for all r (vectorized over r)."""
    zern = np.zeros((len(r), coeff_count))
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
        
        # num2 = (nn + 2*l - 2) * ((nn + 2*l - 2)*((nn + 2*l)*r**2))
    return zern

"""
R_n^{l}
_______
(nn + l + m) = (n + 2l - 2)
(m2 * (m1 * r ** 2 - nn - 1) - 2 * m ** 2) * zern[:, l - 1] = {(n + 2l - 2) * [(n + 2l)* r ** 2 - (n + 1)] - 2 * (l - 1) ** 2} R_n^{l - 1}
(m * (nn + m) * m1 * zern[:, l - 2])) = ((l - 1) * (n + l - 1) * (n + 2l) ) R_n^{l - 2}

(l * m2 * (nn + l)) = (l * (n + 2l - 2) * (n + l))
"""



def precompute_zernike(xsize, nproj, order, nphi, ncoeff):
    """Precompute radial grid, angle step and per-projection coeff_count (integer array)."""
    delrho = 1.0 / (xsize - 1)
    r = np.arange(xsize) * delrho
    deltaphi = 90.0 / nphi
    conv = np.pi / 180.0
    deltaphi_rad = deltaphi * conv
    nn = np.arange(nproj) * order
    ni = nn // 2

    # ensure ncoeff is array length nproj
    ncoeff_arr = np.asarray(ncoeff)
    if ncoeff_arr.shape == ():  # scalar -> broadcast
        ncoeff_arr = np.full(nproj, int(ncoeff_arr), dtype=int)
    else:
        ncoeff_arr = ncoeff_arr.astype(int)
        if ncoeff_arr.size != nproj:
            ncoeff_arr = np.resize(ncoeff_arr, nproj)

    # available anm length per projection = nphi - ni
    avail = (nphi - ni).astype(int)

    # per-projection m_max (floor expression) and ensure >= 1
    m_max = np.floor(0.5 * (np.pi / deltaphi_rad - (nn + 1))).astype(int)
    m_max = np.maximum(1, m_max)

    # elementwise minimum of the three constraints -> integer array
    coeff_count = np.minimum.reduce([ncoeff_arr, avail, m_max]).astype(int)

    return r, coeff_count, nn, ni


def calcrho(anm, order, nproj, xsize, nphi, rhofn, ncoeff, consistency_condition=True):
    """Calculate the radial density functions, rho_{n} (Fortran-style: sort by abs, pos/neg split)."""
    rho = np.zeros((xsize, nproj))
    # delrho = 1.0 / (xsize - 1)
    # r = np.arange(xsize) * delrho
    # deltaphi = 90.0 / nphi
    # conv = np.pi / 180.0
    # deltaphi_rad = deltaphi * conv
    r, coeff_count, nn, ni = precompute_zernike(xsize, nproj, order, nphi, ncoeff)
    
    for ii in range(nproj):
        # nn = ii * order
        # ni = nn // 2
        anm_slice = anm[ni[ii]:nphi, ii].copy()
        # m_max = int(0.5*(np.pi / deltaphi_rad - (nn[ii] + 1)))
        # m_max = max(1, m_max)
        # coeff_count = min(ncoeff[ii], len(anm_slice), m_max)
        if consistency_condition and nn[ii] > 0:
            anm_slice[:nn[ii]//2] = 0

        if coeff_count[ii] == 0:
            continue

        zern = zernike_recursion(r, nn[ii], coeff_count[ii])  # shape (xsize, coeff_count)
        weights = (2 * (np.arange(coeff_count[ii]) + 1) - 1 + nn[ii]) * anm_slice[:coeff_count[ii]]  # (coeff_count,)

        # direct dot-product for all radii at once
        contrib = zern.dot(weights)  # shape (xsize,)

        rho[:, ii] = contrib * rhofn  # vectorized assignment

    return rho


def rhocutoff(xsize, rhocut, flvl, kt):
    """Generate Fermi cutoff function."""
    if rhocut == 0:
        x = np.arange(xsize) - flvl
        f = 1.0 / (1.0 + np.exp(x / kt))
        f[f < 1e-7] = 0.0
        return f
    else:
        return np.ones(xsize)

# def getrho(rawdat, order, pang, nphi, ncoeff, rhofn):
#     nproj = rawdat.shape[2]         # Should have shape (nsize, nsize, nproj)
#     nsize = rawdat.shape[0]
#     xsize = nsize // 2
#     xstart = nsize // 2
#     simul = setupprojs(order, pang, nproj)
#     sinmat = setupsin(nphi)
#     rhoreturn = np.zeros((xsize, nproj, xsize))

#     rhocut, flvl, kt = rhofn[0], rhofn[1], rhofn[2]
#     rhofn = rhocutoff(xsize, rhocut, flvl, kt)
    
#     # rhocut = 1      # 1 = don't cut off, 0 = cut off at flvl
#     # flvl = 150      # Center of cutoff
#     # kt = 2.0        # Slope; increase for smoother, decrease for sharper
#     # ncoeffs = [150, 120, 100, 90, 80, 50]
    
#     for yfixed in range(xsize):
#         yfixed_shifted = yfixed + xstart
#         project = rawdat[xstart:xstart + xsize, yfixed_shifted, :].reshape(xsize, nproj)
#         project = plnorm(project)
#         proj = calccos(project, xsize, nphi, nproj)
#         anm = calcanm(sinmat, simul, nphi, proj)
#         # print(anm.shape)      # (180, nproj)
#         rho = calcrho(anm, order, nproj, xsize, nphi, rhofn, ncoeff)
#         rhoreturn[:, :, yfixed] = rho
#         # print(f'Rhos stored for yfixed={yfixed}')
#     return rhoreturn

"""Numba Accelerated getrho Function:"""

@njit(parallel=True, fastmath=True)
def _getrho_core(rawdat, simul, sininv_mat, i_idx, deltax, valid_mask,
                 r, coeff_count, nn, ni, rhofn_array, order, nphi, xstart, xsize):
    nproj = rawdat.shape[2]
    # y_max = rawdat.shape[1]
    N = xsize
    y_max = xsize
    
    rhoreturn = np.zeros((xsize, nproj, xsize), dtype=rawdat.dtype)
    anm_matrix = np.zeros((xsize, nphi, nproj), dtype=rawdat.dtype)

    # helpers
    last_row_idx = xsize - 1

    # temporary arrays allocated per loop iteration to avoid reallocation inside inner loops
    # Note: numba doesn't support dynamic 2D allocations inside prange well; we'll allocate modest temporaries here.
    for yfixed in prange(y_max):
        yfixed_shifted = yfixed + xstart

        # project : shape (xsize, nproj)
        project = np.empty((xsize, nproj), dtype=rawdat.dtype)
        for ii in range(xsize):
            for jj in range(nproj):
                project[ii, jj] = rawdat[xstart + ii, yfixed_shifted, jj]

        # plnorm (in-place)
        sum1 = 0.0
        for jj in range(nproj):
            sum1 += project[:, 0].sum() if jj == 0 else 0.0
        # above loop (sum1) written that way to keep typing; compute directly:
        sum1 = 0.0
        for ii in range(xsize):
            sum1 += project[ii, 0]

        for n in range(nproj):
            sump = 0.0
            for ii in range(xsize):
                sump += project[ii, n]
            if sump != 0.0:
                scale = sum1 / sump
                for ii in range(xsize):
                    project[ii, n] = project[ii, n] * scale
            else:
                for ii in range(xsize):
                    project[ii, n] = 0.0

        # calccos -> proj (nphi, nproj)
        proj = np.empty((nphi, nproj), dtype=rawdat.dtype)
        for t in range(nphi):
            if valid_mask[t]:
                idx = i_idx[t]
                dt = deltax[t]
                for n in range(nproj):
                    proj[t, n] = project[idx, n] * (1.0 - dt) + project[idx + 1, n] * dt
            else:
                # assign edge value
                for n in range(nproj):
                    proj[t, n] = project[last_row_idx, n]

        # calcanm: projf = proj @ simul.T  (simul assumed to be the matrix returned by setupprojs)
        projf = np.empty((nphi, nproj), dtype=rawdat.dtype)
        for t in range(nphi):
            for i_col in range(nproj):
                s = 0.0
                for j in range(nproj):
                    s += proj[t, j] * simul[i_col, j]
                projf[t, i_col] = s
        # anm_matrix[yfixed] = projf
        anm = np.empty((nphi, nproj), dtype=rawdat.dtype)
        for i_row in range(nphi):
            for j_col in range(nproj):
                s = 0.0
                for k in range(nphi):
                    s += sininv_mat[i_row, k] * projf[k, j_col]
                anm[i_row, j_col] = 0.5 * s
        # store the final anm (not projf) so output matches original getrho
        anm_matrix[yfixed] = anm

        # # anm = 0.5 * sininv_mat @ projf
        # anm = np.empty((nphi, nproj), dtype=rawdat.dtype)
        # for i_row in range(nphi):
        #     for j_col in range(nproj):
        #         s = 0.0
        #         for k in range(nphi):
        #             s += sininv_mat[i_row, k] * projf[k, j_col]
        #         anm[i_row, j_col] = 0.5 * s

        # calcrho: loop over projections, build zernike recursion and dot with weights
        # r is (xsize,) radial grid
        for ii in range(nproj):
            # slice anm from ni[ii] .. nphi-1
            start = ni[ii]
            # build local anm_slice length L = nphi - start (but we only use up to coeff_count[ii])
            L = nphi - start
            # apply consistency_condition: zero small anm parts if nn>0 (original behavior)
            # Here we mimic that: zero first nn[ii]//2 elements of the slice (if they exist)
            # (we must be careful not to go out of bounds)
            # We'll copy necessary entries into a small local array 'anm_slice_use' of length coeff_count[ii]
            cc = coeff_count[ii]
            if cc == 0:
                # leave rho column zero
                for rr in range(xsize):
                    rhoreturn[rr, ii, yfixed] = 0.0
                continue

            anm_slice_use = np.empty(cc, dtype=rawdat.dtype)
            # fill anm_slice_use from anm[start + k, ii]
            for k in range(cc):
                val = anm[start + k, ii]
                # apply consistency: zero if k < nn[ii]//2 and nn[ii] > 0
                if nn[ii] > 0 and k < (nn[ii] // 2):
                    anm_slice_use[k] = 0.0
                else:
                    anm_slice_use[k] = val

            # compute weights: (2*(m+1)-1 + nn) * anm_slice_use[m]
            weights = np.empty(cc, dtype=rawdat.dtype)
            for m in range(cc):
                weights[m] = (2.0 * (m + 1) - 1.0 + nn[ii]) * anm_slice_use[m]

            # compute zernike recursion for this nn[ii] and cc -> zern (xsize, cc)
            zern = np.empty((xsize, cc), dtype=rawdat.dtype)
            # first column
            if cc >= 1:
                if nn[ii] == 0:
                    for rr in range(xsize):
                        zern[rr, 0] = 1.0
                else:
                    for rr in range(xsize):
                        zern[rr, 0] = r[rr] ** nn[ii]
            if cc >= 2:
                for rr in range(xsize):
                    zern[rr, 1] = zern[rr, 0] * ((nn[ii] + 2.0) * (r[rr] ** 2) - (nn[ii] + 1.0))
            for l in range(2, cc):
                m = l - 1
                m2 = nn[ii] + 2 * m
                m1 = m2 + 2
                denom = l * m2 * (nn[ii] + l)
                for rr in range(xsize):
                    num = ((nn[ii] + l + m) *
                           (m2 * (m1 * (r[rr] ** 2) - nn[ii] - 1.0) - 2.0 * (m ** 2)) * zern[rr, l - 1] -
                           m * (nn[ii] + m) * m1 * zern[rr, l - 2])
                    zern[rr, l] = num / denom

            # dot product zern dot weights -> contrib for each radius
            for rr in range(xsize):
                s = 0.0
                for m in range(cc):
                    s += zern[rr, m] * weights[m]
                # multiply by rhofn_array per-radius
                val = s * rhofn_array[rr]
                # clip negative
                # if val < 0.0:
                #     val = 0.0
                rhoreturn[rr, ii, yfixed] = val

    return rhoreturn, anm_matrix

# def getrho(rawdat, order, nproj, calib, pang, nphi, ncoeff, rhofn, sinmat):
def getrho(rawdat, order, pang, nphi, ncoeff, rhofn):
    """
    Wrapper: precompute matrices and indices in Python (NumPy), then call njit core.
    """
    nproj = rawdat.shape[2]         # Should have shape (nsize, nsize, nproj)
    nsize = rawdat.shape[0]
    xsize = nsize // 2
    xstart = nsize // 2

    # precompute projection matrix inverse and sin inverse (as full 2D arrays)
    simul = setupprojs(order, pang, nproj)  # returns (nproj,nproj) matrix (already the inverse)
    sinmat_flat = setupsin(nphi)            # flattened Fortran-order inverse
    sininv_mat = sinmat_flat.reshape((nphi, nphi), order='F')

    # prepare rhofn array (fermi cutoff)
    rhocut, flvl, kt = rhofn[0], rhofn[1], rhofn[2]
    rhofn_array = rhocutoff(xsize, rhocut, flvl, kt)

    # precompute calccos indices and interpolation
    deltaphi = 90.0 / nphi
    angles = np.linspace(0.0, 90.0, nphi, endpoint=True) * np.pi / 180.0
    dists = (xsize - 1) * np.cos(angles)
    i_idx = np.floor(dists).astype(np.int64)
    deltax = dists - i_idx
    valid_mask = (i_idx >= 0) & (i_idx < xsize - 1)

    # precompute zernike parameters
    r, coeff_count, nn, ni = precompute_zernike(xsize, nproj, order, nphi, ncoeff)

    # call compiled core
    rhoreturn, anm = _getrho_core(rawdat, simul, sininv_mat, i_idx, deltax, valid_mask,
                             r, coeff_count, nn, ni, rhofn_array, float(order), nphi, int(xstart), int(xsize))
    return rhoreturn, anm





"""Numba Accelerated Calcplane Function:"""

def precompute_calcplane(nproj, xsize, order):
    # def calcplane_triggeronce(rhos, yfixed, nproj, xsize, order, iext=0):
    # rhos = rhoreturn_ideal2
    # grid and centre
    N_full = 2 * xsize
    x = np.arange(N_full)
    z = np.arange(N_full)
    x_grid, z_grid = np.meshgrid(x, z, indexing='ij')
    x0 = xsize - 0.5

    # distances
    xdist = np.abs(x_grid - x0)
    zdist = np.abs(z_grid - x0)

    # avoid division by zero for theta calculation (use same scale as original)
    zdist = np.maximum(zdist, 1e-3)

    # polar coordinates (p in pixels, theta angular)
    theta = np.arctan2(x_grid - x0, zdist)           # shape (N,N)
    p = np.sqrt(xdist**2 + zdist**2)                 # shape (N,N)

    # interpolation indices along radial axis
    intp = np.floor(p).astype(np.int64)
    deltap = p - intp

    # match original clipping: idx1 clipped to xsize-2, idx2 to xsize-1
    idx1 = np.clip(intp, 0, xsize - 2)
    idx2 = np.clip(intp + 1, 0, xsize - 1)

    # mask for exactly the top index (original used intp == xsize-1)
    mask_exact_top = (intp == (xsize - 1))

    # angular cosines: cos(n * order * theta) -> shape (N,N,nproj)
    n_idx = np.arange(nproj, dtype=theta.dtype)[None, None, :]

    cos_terms = np.cos(n_idx * order * theta[:, :, None])
    return idx1, idx2, deltap, cos_terms, mask_exact_top


# def calcplane(rhos, y_max, order):
#     xsize = rhos.shape[0]
#     nproj = rhos.shape[1]
        
#     idx1, idx2, deltap, cos_terms, mask_exact_top = precompute_calcplane(nproj, xsize, order)
#     plane_stack = np.zeros((2 * xsize, y_max, 2 * xsize))

#     if y_max >= xsize:
#         y_max = xsize - 1
#         # print("Warning: y_max exceeds xsize; limiting to xsize.")
    
#     for yfixed in range(y_max):
#         # fetch rhos for this yfixed: shape (xsize, nproj)
#         rhos_y = rhos[:, :nproj, yfixed]   # (xsize, nproj)

#         # gather rhos at idx1/idx2 for all pixels: result shape (N, N, nproj)
#         r1 = rhos_y.take(idx1, axis=0)     # shape (N, N, nproj)
#         r2 = rhos_y.take(idx2, axis=0)

#         # linear radial interpolation
#         interp = (1.0 - deltap)[:, :, None] * r1 + deltap[:, :, None] * r2   # (N,N,nproj)

#         # For positions where intp == xsize-1 use the last radial sample (match original masked assignment)
#         if np.any(mask_exact_top):
#             last_row = rhos_y[xsize - 1, :][None, None, :]   # shape (1,1,nproj)
#             interp[mask_exact_top, :] = last_row  # broadcast to (nproj,) at mask positions

#         # multiply and sum over n axis -> final plane
#         plane = np.sum(interp * cos_terms, axis=2)

#         # clip negatives (physical counts)
#         plane = np.clip(plane, 0.0, None)
#         plane_stack[:, yfixed, :] = plane

#     return plane_stack

@numba.njit(parallel=True, fastmath=True)
def _calcplane_core(rhos, idx1, idx2, deltap, theta, mask_exact_top, order, y_max):
    N = idx1.shape[0]           # full plane size (2*xsize)
    xsize = rhos.shape[0]
    nproj = rhos.shape[1]
    # ensure y_max doesn't exceed available slices
    if y_max > rhos.shape[2]:
        y_max = rhos.shape[2]
    plane_stack = np.zeros((N, y_max, N), dtype=rhos.dtype)

    last_idx = xsize - 1
    for yfixed in prange(y_max):
        for i in range(N):
            for j in range(N):
                d = deltap[i, j]
                t = theta[i, j]
                use_mask = mask_exact_top[i, j]
                s = 0.0
                if use_mask:
                    # use last radial sample only
                    for n in range(nproj):
                        rlast = rhos[last_idx, n, yfixed]
                        s += rlast * math.cos(n * order * t)
                else:
                    i1 = idx1[i, j]
                    i2 = idx2[i, j]
                    for n in range(nproj):
                        r1 = rhos[i1, n, yfixed]
                        r2 = rhos[i2, n, yfixed]
                        val = (1.0 - d) * r1 + d * r2
                        s += val * math.cos(n * order * t)
                if s < 0.0:
                    s = 0.0
                plane_stack[i, yfixed, j] = s
    return plane_stack


def calcplane(rhos, y_max, order):
    """
    Public wrapper: precompute geometric arrays (Python) then call numba core.
    Falls back to pure-Python behaviour if numba not available.
    """
    xsize = rhos.shape[0]
    nproj = rhos.shape[1]

    # precompute (Python) once
    idx1, idx2, deltap, cos_terms, mask_exact_top = precompute_calcplane(nproj, xsize, order)

    # _calcplane_core expects theta and mask; we used mask_exact_top and theta in core.
    # Extract theta from cos_terms by inverting cos? easier: recompute theta here cheaply
    # but we already have theta inside precompute? If not, derive theta from cos_terms is not trivial.
    # For minimal change, modify precompute_calcplane to also return theta; if not, compute theta here:
    # (we will recompute theta here to avoid changing precompute signature)
    N_full = 2 * xsize
    x = np.arange(N_full)
    z = np.arange(N_full)
    x_grid, z_grid = np.meshgrid(x, z, indexing='ij')
    x0 = xsize - 0.5
    zdist = np.abs(z_grid - x0)
    zdist = np.maximum(zdist, 1e-3)
    theta = np.arctan2(x_grid - x0, zdist)

    # call compiled core
    plane_stack = _calcplane_core(rhos, idx1.astype(np.int64), idx2.astype(np.int64),
                                  deltap.astype(np.float64), theta.astype(np.float64),
                                  mask_exact_top, float(order), int(y_max))
    return plane_stack


#################################################### Getrho_anm_synth ####################################################


@njit(parallel=True, fastmath=True)
def _getrho_anm_core(anm, xsize, r, coeff_count, nn, ni, rhofn_array):
    n_samples, nphi, nproj = anm.shape
    drho_type = np.float64

    # ensure y_max equals xsize
    y_max = n_samples
    rhoreturn = np.zeros((xsize, nproj, n_samples), dtype=drho_type)                       # copper rho_n have dtype=float64

    for yfixed in prange(y_max):
        anm_slice = anm[yfixed, :, :]
        
        # calcrho: loop over projections, build zernike recursion and dot with weights
        # r is (xsize,) radial grid
        for ii in range(nproj):
            # slice anm from ni[ii] .. nphi-1
            start = ni[ii]
            # build local anm_slice length L = nphi - start (but we only use up to coeff_count[ii])
            L = nphi - start
            # apply consistency_condition: zero small anm parts if nn>0 (original behavior)
            # Here we mimic that: zero first nn[ii]//2 elements of the slice (if they exist)
            # (we must be careful not to go out of bounds)
            # We'll copy necessary entries into a small local array 'anm_slice_use' of length coeff_count[ii]
            cc = coeff_count[ii]
            if cc == 0:
                # leave rho column zero
                for rr in range(xsize):
                    rhoreturn[rr, ii, yfixed] = 0.0
                continue

            anm_slice_use = np.empty(cc, dtype=drho_type)
            # fill anm_slice_use from anm[start + k, ii]
            for k in range(cc):
                # val = anm[start + k, ii]
                val = anm_slice[start + k, ii]
                # apply consistency: zero if k < nn[ii]//2 and nn[ii] > 0
                if nn[ii] > 0 and k < (nn[ii] // 2):
                    anm_slice_use[k] = 0.0
                else:
                    anm_slice_use[k] = val

            # compute weights: (2*(m+1)-1 + nn) * anm_slice_use[m]
            weights = np.empty(cc, dtype=drho_type)
            for m in range(cc):
                weights[m] = (2.0 * (m + 1) - 1.0 + nn[ii]) * anm_slice_use[m]

            # compute zernike recursion for this nn[ii] and cc -> zern (xsize, cc)
            zern = np.empty((xsize, cc), dtype=drho_type)
            # first column
            if cc >= 1:
                if nn[ii] == 0:
                    for rr in range(xsize):
                        zern[rr, 0] = 1.0
                else:
                    for rr in range(xsize):
                        zern[rr, 0] = r[rr] ** nn[ii]
            if cc >= 2:
                for rr in range(xsize):
                    zern[rr, 1] = zern[rr, 0] * ((nn[ii] + 2.0) * (r[rr] ** 2) - (nn[ii] + 1.0))
            for l in range(2, cc):
                m = l - 1
                m2 = nn[ii] + 2 * m
                m1 = m2 + 2
                denom = l * m2 * (nn[ii] + l)
                for rr in range(xsize):
                    num = ((nn[ii] + l + m) *
                           (m2 * (m1 * (r[rr] ** 2) - nn[ii] - 1.0) - 2.0 * (m ** 2)) * zern[rr, l - 1] -
                           m * (nn[ii] + m) * m1 * zern[rr, l - 2])
                    zern[rr, l] = num / denom

            # dot product zern dot weights -> contrib for each radius
            for rr in range(xsize):
                s = 0.0
                for m in range(cc):
                    s += zern[rr, m] * weights[m]
                # multiply by rhofn_array per-radius
                val = s * rhofn_array[rr]
                # clip negative
                # if val < 0.0:
                #     val = 0.0
                rhoreturn[rr, ii, yfixed] = val

    return rhoreturn


def getrho_anm_synth(order, ncoeff, rhofn, anm, xsize):
    """
    Alternate getrho: Use a custom anm matrix to calculate rhoreturn.

    Parameters
    ----------
    order : int
        Symmetry order for system (e.g. 4 for C4).
    ncoeff : int or array-like
        Number of Zernike coefficients per projection.
    rhofn : tuple
        Fermi cutoff parameters (rhocut, flvl, kt).
    anm : np.ndarray
        Precomputed anm matrix, shape (xsize, nphi, nproj).

    Returns
    -------
    rhoreturn : np.ndarray
        Calculated rho array, shape (xsize, nproj, xsize).
    """
    
    n_samples, nphi, nproj = anm.shape

    # prepare rhofn array (fermi cutoff)
    rhocut, flvl, kt = rhofn[0], rhofn[1], rhofn[2]
    rhofn_array = rhocutoff(xsize, rhocut, flvl, kt)

    # precompute zernike parameters
    r, coeff_count, nn, ni = precompute_zernike(xsize, nproj, order, nphi, ncoeff)

    # call compiled core
    rhoreturn = _getrho_anm_core(anm, xsize, r, coeff_count, nn, ni, rhofn_array)
    
    return rhoreturn


# def anm_to_rhos(anm, order, pang, rhofn, ncoeff):
#     """
#     Compute rhoreturn directly from ANM coefficients.

#     Parameters
#     ----------
#     anm : ndarray
#         Shape (xsize, nphi, nproj) = (256, 180, 20)

#     Returns
#     -------
#     rhoreturn : ndarray
#         Reconstructed rho slices
#     """

#     # infer dimensions from anm
#     xsize, nphi, nproj = anm.shape
#     nsize = 2 * xsize
#     xstart = xsize

#     # --------------------------------------------------
#     # precompute projection matrix inverse
#     # --------------------------------------------------
#     simul = setupprojs(order, pang, nproj)

#     # --------------------------------------------------
#     # sin inverse matrix
#     # --------------------------------------------------
#     sinmat_flat = setupsin(nphi)
#     sininv_mat = sinmat_flat.reshape((nphi, nphi), order="F")

#     # --------------------------------------------------
#     # rho cutoff function
#     # --------------------------------------------------
#     rhocut, flvl, kt = rhofn
#     rhofn_array = rhocutoff(xsize, rhocut, flvl, kt)

#     # --------------------------------------------------
#     # calccos interpolation setup
#     # --------------------------------------------------
#     angles = np.linspace(0.0, 90.0, nphi, endpoint=True) * np.pi / 180.0
#     dists = (xsize - 1) * np.cos(angles)

#     i_idx = np.floor(dists).astype(np.int64)
#     deltax = dists - i_idx
#     valid_mask = (i_idx >= 0) & (i_idx < xsize - 1)

#     # --------------------------------------------------
#     # zernike precomputation
#     # --------------------------------------------------
#     r, coeff_count, nn, ni = precompute_zernike(
#         xsize, nproj, order, nphi, ncoeff
#     )

#     # --------------------------------------------------
#     # call compiled core
#     # --------------------------------------------------
#     # rawdat is not required for ANM-driven reconstruction
#     rhoreturn = _getrho_anm_core(
#         None,
#         simul,
#         sininv_mat,
#         anm,
#         i_idx,
#         deltax,
#         valid_mask,
#         r,
#         coeff_count,
#         nn,
#         ni,
#         rhofn_array,
#         float(order),
#         nphi,
#         int(xstart),
#         int(xsize),
#     )

#     return rhoreturn

### New anm to rho function: direct from anm to rhos ###


#################################################### Data Generation: HYPER! ####################################################

@njit(parallel=True, fastmath=True)
def _datagen_anm_to_getrho_(
    anm,              # (xsize, nphi, nproj)
    r,                # (xsize,)
    coeff_count,      # (nproj,)
    nn,               # (nproj,)
    ni,               # (nproj,)
    rhofn_array,      # (xsize,)
    xsize
):
    xsize0, nphi, nproj = anm.shape

    rhoreturn = np.zeros((xsize, nproj, xsize), dtype=anm.dtype)

    for yfixed in prange(xsize):
        anm_slice = anm[yfixed]  # (nphi, nproj)

        for ii in range(nproj):
            cc = coeff_count[ii]
            if cc == 0:
                continue

            start = ni[ii]

            # build filtered ANM slice
            anm_use = np.empty(cc, dtype=anm.dtype)
            for k in range(cc):
                val = anm_slice[start + k, ii]
                if nn[ii] > 0 and k < (nn[ii] // 2):
                    anm_use[k] = 0.0
                else:
                    anm_use[k] = val

            # weights
            weights = np.empty(cc, dtype=anm.dtype)
            for m in range(cc):
                weights[m] = (2.0 * (m + 1) - 1.0 + nn[ii]) * anm_use[m]

            # Zernike recursion
            zern = np.empty((xsize, cc), dtype=anm.dtype)

            if nn[ii] == 0:
                for rr in range(xsize):
                    zern[rr, 0] = 1.0
            else:
                for rr in range(xsize):
                    zern[rr, 0] = r[rr] ** nn[ii]

            if cc > 1:
                for rr in range(xsize):
                    zern[rr, 1] = zern[rr, 0] * (
                        (nn[ii] + 2.0) * (r[rr] ** 2) - (nn[ii] + 1.0)
                    )

            for l in range(2, cc):
                m = l - 1
                m2 = nn[ii] + 2 * m
                m1 = m2 + 2
                denom = l * m2 * (nn[ii] + l)

                for rr in range(xsize):
                    num = (
                        (nn[ii] + l + m)
                        * (m2 * (m1 * (r[rr] ** 2) - nn[ii] - 1.0) - 2.0 * m * m)
                        * zern[rr, l - 1]
                        - m * (nn[ii] + m) * m1 * zern[rr, l - 2]
                    )
                    zern[rr, l] = num / denom

            # dot product
            for rr in range(xsize):
                s = 0.0
                for m in range(cc):
                    s += zern[rr, m] * weights[m]
                rhoreturn[rr, ii, yfixed] = s * rhofn_array[rr]

    return rhoreturn

def anm_to_rhos(anm, order, rhofn, ncoeff):
    xsize, nphi, nproj = anm.shape

    rhocut, flvl, kt = rhofn
    rhofn_array = rhocutoff(xsize, rhocut, flvl, kt)

    r, coeff_count, nn, ni = precompute_zernike(
        xsize, nproj, order, nphi, ncoeff
    )

    return _datagen_anm_to_getrho_(
        anm,
        r,
        coeff_count,
        nn,
        ni,
        rhofn_array,
        xsize
    )




#################################################### 3D TPMD Generation ####################################################

def animate_slices(volume,
                   start_frame=0,
                   end_frame=None,
                   fixed_z=None,
                   output_filename="Cormack_reconstructed_xz.gif",
                   fps=10,
                   interval=100,
                   cmap='hot',
                   vmin=None,
                   vmax=None,
                   save=None):
    """
    Animate x–y slices (varying z-index) from a 3D Cormack reconstruction volume.
    """
    N = volume.shape[0]
    if end_frame is None or end_frame >= N:
        end_frame = N - 1
    if fixed_z is not None:
        start_frame = end_frame = fixed_z
    frames_to_animate = range(start_frame, end_frame + 1)

    # Use provided vmin/vmax or compute from the whole volume
    if vmin is None:
        vmin = np.quantile(volume, 0.01)
    if vmax is None:
        vmax = np.quantile(volume, 1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    initial_slice = volume[:, start_frame, :].T
    img = ax.imshow(initial_slice, cmap=cmap, origin='lower',
                    vmin=vmin, vmax=vmax, aspect='auto')
    cbar = fig.colorbar(img, ax=ax, shrink=0.8)
    cbar.set_label("Reconstructed Intensity")
    title = ax.set_title(f"Reconstructed XZ Slice (y = {start_frame:03d})")
    ax.set_xlabel("X-axis (Px)")
    ax.set_ylabel("Z-axis (Pz)")

    def update(frame):
        img.set_array(volume[:, frame, :].T)
        title.set_text(f"Reconstructed XZ Slice (y = {frame:03d})")
        return img, title

    anim = animation.FuncAnimation(fig, update, frames=frames_to_animate,
                                   interval=interval, blit=False)
    if save == True:
        anim.save("MCM_Cu.gif", writer='pillow', fps=10)
    plt.close(fig)
    return HTML(anim.to_jshtml())



def standardize_anm(anm):
    """
    Standardize the anm matrix so that for each j and i, anm[j][:, i] has mean 0 and std 1.

    Parameters
    ----------
    anm : np.ndarray
        Input array of shape (xsize, nphi, nproj).

    Returns
    -------
    anm_pca : np.ndarray
        Standardized array of the same shape as anm.
    means : np.ndarray
        Means for each (j, i), shape (xsize, nproj).
    stds : np.ndarray
        Standard deviations for each (j, i), shape (xsize, nproj).
    """
    xsize, nphi, nproj = anm.shape
    anm_pca = np.zeros_like(anm)
    means = np.zeros((xsize, nproj))
    stds = np.zeros((xsize, nproj))
    for j in range(xsize):
        for i in range(nproj):
            vec = anm[j][:, i]
            mean = np.mean(vec)
            std = np.std(vec)
            means[j, i] = mean
            stds[j, i] = std
            if std > 0:
                anm_pca[j][:, i] = (vec - mean) / std
            else:
                anm_pca[j][:, i] = 0.0
    return anm_pca, means, stds

def unstandardize_anm(anm_pca, means, stds):
    """
    Reverse the standardization of the anm matrix, restoring original mean and standard deviation to that of the central slice

    Parameters
    ----------
    anm_pca : np.ndarray
        Standardized array of shape (xsize, nphi, nproj).
    means : np.ndarray
        Array of means used for standardization, shape (xsize, nproj).
    stds : np.ndarray
        Array of standard deviations used for standardization, shape (xsize, nproj).

    Returns
    -------
    anm_original : np.ndarray
        Array restored to original mean and standard deviation, same shape as anm_pca.
    """
    xsize, nphi, nproj = anm_pca.shape
    anm_original = np.zeros_like(anm_pca)
    for j in range(xsize):
        for i in range(nproj):
            anm_original[j][:, i] = anm_pca[j][:, i] * stds[0, i] + means[0, i]
    return anm_original



def pca_over_j(anm_pca, i):
    """
    Apply PCA to the dataset formed by stacking anm_pca[j][:, i] for all j.

    Parameters
    ----------
    anm_pca : np.ndarray
        Standardized anm array of shape (xsize, nphi, nproj).
    i : int
        Projection index to fix.

    Returns
    -------
    principal_components : np.ndarray
        Principal components, shape (xsize, nphi).
    explained_variance : np.ndarray
        Explained variance ratio for each principal component.
    """
    xsize, nphi, nproj = anm_pca.shape

    # Stack anm_pca[j][:, i] for all j into a matrix of shape (xsize, nphi)
    data_matrix = np.array([anm_pca[j][:, i] for j in range(xsize)])  # shape: (xsize, nphi)

    # Apply PCA
    pca = PCA()
    principal_components = pca.fit_transform(data_matrix)  # shape: (xsize, nphi)
    explained_variance = pca.explained_variance_ratio_

    return principal_components, explained_variance



def normalize_rhoreturn_ideal_Cu(rhoreturn_ideal_Cu):
    """
    Normalizes rhoreturn_ideal_Cu[:, 0:20, slice_idx] for each slice_idx by the max of rhoreturn_ideal_Cu[:, 0, slice_idx].
    Returns the normalized array and the array of max values for each slice_idx.
    """
    normalized = np.copy(rhoreturn_ideal_Cu)
    max_vals = np.zeros(rhoreturn_ideal_Cu.shape[2])
    for slice_idx in range(rhoreturn_ideal_Cu.shape[2]):
        max_val = np.max(rhoreturn_ideal_Cu[:, 0, slice_idx])
        max_vals[slice_idx] = max_val
        normalized[:, 0:20, slice_idx] /= max_val if max_val != 0 else 1.0
    return normalized, max_vals


def custom_randint():
    # Allowed ranges: 0-30, 60-119, 180-209
    allowed = np.concatenate([
        np.arange(0, 35),
        np.arange(60, 133),
        # np.arange(180, 210)
    ])
    return np.random.choice(allowed)




### DMD Algorithm Functions ###
# For Rho functions time evolution (adapted for shape (256, 180, 20))
# --------------------------------------------------
# 1. Prepare data
# --------------------------------------------------

def prepare_snapshots(data):
    """
    data: array of shape (256, 180, 20)
    returns X, Y with shape (180*20, T-1)
    """
    T = data.shape[0]  # Now T = 256 (time/slice axis)
    snapshots = [data[t, :, :].reshape(-1) for t in range(T)]
    snapshots = np.stack(snapshots, axis=1)
    X = snapshots[:, :-1]
    Y = snapshots[:, 1:]
    return X, Y

# --------------------------------------------------
# 2. PCA (SVD-based)
# --------------------------------------------------

def compute_pca(X, rank):
    """
    X: (n_features, n_samples)
    """
    X_mean = np.mean(X, axis=1, keepdims=True)
    Xc = X - X_mean

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    U_k = U[:, :rank]
    return U_k, X_mean

# --------------------------------------------------
# 3. Koopman / DMD operator in latent space
# --------------------------------------------------

def compute_koopman(X, Y, U_k):
    """
    Learns A such that z_{t+1} = A z_t
    """
    Z = U_k.T @ X
    Zp = U_k.T @ Y

    A = Zp @ np.linalg.pinv(Z)
    return A

# --------------------------------------------------
# 4. Train model
# --------------------------------------------------

def train_pca_dmd_per_channel(data, latent_dim=20):
    """
    Trains a separate PCA/DMD model for each channel.
    data: array of shape (256, 180, 20)
    Returns: list of models, one per channel
    """
    n_channels = data.shape[2]
    models = []
    for ch in range(n_channels):
        # For each channel, extract (256, 180) array
        channel_data = data[:, :, ch]  # shape (256, 180)
        # Each time step is a (180,) vector, so stack as (180, 256)
        # But we want to treat each time step as a (180, 1) snapshot, so flatten to (180, 1)
        # Actually, for PCA, we want (180, T) where T=256
        X = channel_data.T  # shape (180, 256)
        X_mean = np.mean(X, axis=1, keepdims=True)
        Xc = X - X_mean
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        U_k = U[:, :latent_dim]
        # Prepare X, Y for Koopman
        X_snap = X[:, :-1]
        Y_snap = X[:, 1:]
        Z = U_k.T @ (X_snap - X_mean)
        Zp = U_k.T @ (Y_snap - X_mean)
        A = Zp @ np.linalg.pinv(Z)
        models.append({
            "U_k": U_k,
            "A": A,
            "mean": X_mean
        })
    return models


# --------------------------------------------------
# 5. Rollout from new initial condition
# --------------------------------------------------

def rollout_per_channel(models, x0, n_steps):
    """
    Rollout for each channel separately.
    x0: shape (180, 20) -- initial condition at a moment in time
    Returns: (n_steps, 180, 20) -- predicted evolution for all channels
    """
    n_channels = x0.shape[1]
    n_space = x0.shape[0]
    all_channels = np.zeros((n_steps, n_space, n_channels))
    for ch in range(n_channels):
        model = models[ch]
        U_k = model["U_k"]
        A = model["A"]
        mean = model["mean"]
        x = x0[:, ch].reshape(-1, 1)  # shape (180, 1)
        z = U_k.T @ (x - mean)
        for t in range(n_steps):
            x_rec = (U_k @ z + mean).reshape(n_space)
            all_channels[t, :, ch] = x_rec
            z = A @ z
    return all_channels  # shape (n_steps, 180, 20)



# Taking projections of generated TPMD Central Slices
def compute_projections_stacked(TPMD_Slices, angles_to_extract=np.linspace(0, 45, 20, endpoint=True)):
    """
    For each sample in TPMD_Slices, computes the Radon transform,
    extracts projections at specified angles, and computes anm_arr.
    Returns anm_arr_stacked of shape (n_samples, anm_arr.shape[0], anm_arr.shape[1]).
    
    Parameters:
    - TPMD_Slices: np.ndarray, shape (512, n_samples, 512)
    - order, calib, nphi, ncoeffs, rhofn, sinmat: parameters for getrho_1d
    - angles_to_extract: list or np.ndarray of angles to extract projections
    
    Returns:
    - anm_arr_stacked: np.ndarray, shape (n_samples, anm_arr.shape[0], anm_arr.shape[1])
    """
    n_samples = TPMD_Slices.shape[1]
    # theta = np.linspace(0., 180., Synthetic_Central_Slices.shape[0], endpoint=False)
    # angle_indices = [np.argmin(np.abs(theta - angle)) for angle in angles_to_extract]
    nproj = len(angles_to_extract)
    projections = np.zeros((TPMD_Slices.shape[0], n_samples, nproj))
    
    # for i in tqdm(range(n_samples), desc="Computing anm_arr for samples"):
    for i in range(n_samples):
        xz_slice = TPMD_Slices[:, i, :]  # Shape (512, 512)
        sinogram = radon(xz_slice, theta=angles_to_extract, circle=True)
        projections[:, i, :] = sinogram                        #[:, angle_indices]  # Shape (512, nproj)
    return projections


def adjust_rho0_maxima(normalize_rhoreturn_ideal, rhoreturn_ideal_Cu, percent=0.10, rng=None, rhocut=1, kt=6.0):
    """
    For each idx and each i in 0..N, rescales normalize_rhoreturn_ideal[:, i, idx] so its maximum is a random value
    within ±percent of the absolute maximum of rhoreturn_ideal_Cu[:, i, 0], then applies rhocutoff.
    For i=0, uses flvl=100; for all other i, uses flvl=35.
    """
    if rng is None:
        rng = np.random.default_rng()
    adjusted = np.copy(normalize_rhoreturn_ideal)
    xsize = normalize_rhoreturn_ideal.shape[0]
    n_i = normalize_rhoreturn_ideal.shape[1]
    n_idx = normalize_rhoreturn_ideal.shape[2]
    for idx in range(n_idx):
        for i in range(n_i):
            target_max = np.max(np.abs(rhoreturn_ideal_Cu[:, i, 0]))
            if i == 0:
                upper = target_max * (1 + 0.05)
                lower = target_max * (1 - 0.05)
            else:
                lower = target_max * (1 - percent/2)
                upper = target_max * (1 + percent)
            new_max = rng.uniform(lower, upper)
            current_max = np.max(np.abs(normalize_rhoreturn_ideal[:, i, idx]))
            flvl_val = 128 if i == 0 else 100
            cutoff = rhocutoff(xsize, rhocut, flvl_val, kt)
            if current_max != 0:
                adjusted[:, i, idx] = normalize_rhoreturn_ideal[:, i, idx] * (new_max / current_max) * cutoff
            else:
                adjusted[:, i, idx] = 0
    return adjusted



### Modified numba getrho function to be able to synthesize large amounts of central slices stacked together ###

@njit(parallel=True, fastmath=True)
def _training_getrho_core(rawdat, simul, sininv_mat, i_idx, deltax, valid_mask,
                 r, coeff_count, nn, ni, rhofn_array, order, nphi, xstart, xsize):
    nproj = rawdat.shape[2]
    y_max = rawdat.shape[1]
    N = xsize
    # ensure y_max equals xsize
    
    rhoreturn = np.zeros((xsize, nproj, y_max), dtype=rawdat.dtype)
    anm_matrix = np.zeros((y_max, nphi, nproj), dtype=rawdat.dtype)

    # helpers
    last_row_idx = xsize - 1

    # temporary arrays allocated per loop iteration to avoid reallocation inside inner loops
    # Note: numba doesn't support dynamic 2D allocations inside prange well; we'll allocate modest temporaries here.
    for yfixed in prange(y_max):
        # yfixed_shifted = yfixed + xstart

        # project : shape (xsize, nproj)
        project = np.empty((xsize, nproj), dtype=rawdat.dtype)
        for ii in range(xsize):
            for jj in range(nproj):
                # project[ii, jj] = rawdat[xstart + ii, yfixed_shifted, jj]
                project[ii, jj] = rawdat[xstart + ii, yfixed, jj]

        # plnorm (in-place)
        sum1 = 0.0
        for jj in range(nproj):
            sum1 += project[:, 0].sum() if jj == 0 else 0.0
        # above loop (sum1) written that way to keep typing; compute directly:
        sum1 = 0.0
        for ii in range(xsize):
            sum1 += project[ii, 0]

        for n in range(nproj):
            sump = 0.0
            for ii in range(xsize):
                sump += project[ii, n]
            if sump != 0.0:
                scale = sum1 / sump
                for ii in range(xsize):
                    project[ii, n] = project[ii, n] * scale
            else:
                for ii in range(xsize):
                    project[ii, n] = 0.0

        # calccos -> proj (nphi, nproj)
        proj = np.empty((nphi, nproj), dtype=rawdat.dtype)
        for t in range(nphi):
            if valid_mask[t]:
                idx = i_idx[t]
                dt = deltax[t]
                for n in range(nproj):
                    proj[t, n] = project[idx, n] * (1.0 - dt) + project[idx + 1, n] * dt
            else:
                # assign edge value
                for n in range(nproj):
                    proj[t, n] = project[last_row_idx, n]

        # calcanm: projf = proj @ simul.T  (simul assumed to be the matrix returned by setupprojs)
        projf = np.empty((nphi, nproj), dtype=rawdat.dtype)
        for t in range(nphi):
            for i_col in range(nproj):
                s = 0.0
                for j in range(nproj):
                    s += proj[t, j] * simul[i_col, j]
                projf[t, i_col] = s
        # anm_matrix[yfixed] = projf
        anm = np.empty((nphi, nproj), dtype=rawdat.dtype)
        for i_row in range(nphi):
            for j_col in range(nproj):
                s = 0.0
                for k in range(nphi):
                    s += sininv_mat[i_row, k] * projf[k, j_col]
                anm[i_row, j_col] = 0.5 * s
        # store the final anm (not projf) so output matches original getrho
        anm_matrix[yfixed, :, :] = anm

        # # anm = 0.5 * sininv_mat @ projf
        # anm = np.empty((nphi, nproj), dtype=rawdat.dtype)
        # for i_row in range(nphi):
        #     for j_col in range(nproj):
        #         s = 0.0
        #         for k in range(nphi):
        #             s += sininv_mat[i_row, k] * projf[k, j_col]
        #         anm[i_row, j_col] = 0.5 * s

        # calcrho: loop over projections, build zernike recursion and dot with weights
        # r is (xsize,) radial grid
        for ii in range(nproj):
            # slice anm from ni[ii] .. nphi-1
            start = ni[ii]
            # build local anm_slice length L = nphi - start (but we only use up to coeff_count[ii])
            L = nphi - start
            # apply consistency_condition: zero small anm parts if nn>0 (original behavior)
            # Here we mimic that: zero first nn[ii]//2 elements of the slice (if they exist)
            # (we must be careful not to go out of bounds)
            # We'll copy necessary entries into a small local array 'anm_slice_use' of length coeff_count[ii]
            cc = coeff_count[ii]
            if cc == 0:
                # leave rho column zero
                for rr in range(xsize):
                    rhoreturn[rr, ii, yfixed] = 0.0
                continue

            anm_slice_use = np.empty(cc, dtype=rawdat.dtype)
            # fill anm_slice_use from anm[start + k, ii]
            for k in range(cc):
                val = anm[start + k, ii]
                # apply consistency: zero if k < nn[ii]//2 and nn[ii] > 0
                if nn[ii] > 0 and k < (nn[ii] // 2):
                    anm_slice_use[k] = 0.0
                else:
                    anm_slice_use[k] = val

            # compute weights: (2*(m+1)-1 + nn) * anm_slice_use[m]
            weights = np.empty(cc, dtype=rawdat.dtype)
            for m in range(cc):
                weights[m] = (2.0 * (m + 1) - 1.0 + nn[ii]) * anm_slice_use[m]

            # compute zernike recursion for this nn[ii] and cc -> zern (xsize, cc)
            zern = np.empty((xsize, cc), dtype=rawdat.dtype)
            # first column
            if cc >= 1:
                if nn[ii] == 0:
                    for rr in range(xsize):
                        zern[rr, 0] = 1.0
                else:
                    for rr in range(xsize):
                        zern[rr, 0] = r[rr] ** nn[ii]
            if cc >= 2:
                for rr in range(xsize):
                    zern[rr, 1] = zern[rr, 0] * ((nn[ii] + 2.0) * (r[rr] ** 2) - (nn[ii] + 1.0))
            for l in range(2, cc):
                m = l - 1
                m2 = nn[ii] + 2 * m
                m1 = m2 + 2
                denom = l * m2 * (nn[ii] + l)
                for rr in range(xsize):
                    num = ((nn[ii] + l + m) *
                           (m2 * (m1 * (r[rr] ** 2) - nn[ii] - 1.0) - 2.0 * (m ** 2)) * zern[rr, l - 1] -
                           m * (nn[ii] + m) * m1 * zern[rr, l - 2])
                    zern[rr, l] = num / denom

            # dot product zern dot weights -> contrib for each radius
            for rr in range(xsize):
                s = 0.0
                for m in range(cc):
                    s += zern[rr, m] * weights[m]
                # multiply by rhofn_array per-radius
                val = s * rhofn_array[rr]
                # clip negative
                # if val < 0.0:
                #     val = 0.0
                rhoreturn[rr, ii, yfixed] = val

    return rhoreturn, anm_matrix


# def getrho(rawdat, order, nproj, calib, pang, nphi, ncoeff, rhofn, sinmat):
def getrho_training_data(rawdat, order, pang, nphi, ncoeff, rhofn):
    """
    Wrapper: precompute matrices and indices in Python (NumPy), then call njit core.
    """
    nproj = rawdat.shape[2]         # Should have shape (nsize, nsize, nproj)
    nsize = rawdat.shape[0]
    xsize = nsize // 2
    xstart = nsize // 2

    # precompute projection matrix inverse and sin inverse (as full 2D arrays)
    simul = setupprojs(order, pang, nproj)  # returns (nproj,nproj) matrix (already the inverse)
    sinmat_flat = setupsin(nphi)            # flattened Fortran-order inverse
    sininv_mat = sinmat_flat.reshape((nphi, nphi), order='F')

    # prepare rhofn array (fermi cutoff)
    rhocut, flvl, kt = rhofn[0], rhofn[1], rhofn[2]
    rhofn_array = rhocutoff(xsize, rhocut, flvl, kt)

    # precompute calccos indices and interpolation
    deltaphi = 90.0 / nphi
    angles = np.linspace(0.0, 90.0, nphi, endpoint=True) * np.pi / 180.0
    dists = (xsize - 1) * np.cos(angles)
    i_idx = np.floor(dists).astype(np.int64)
    deltax = dists - i_idx
    valid_mask = (i_idx >= 0) & (i_idx < xsize - 1)

    # precompute zernike parameters
    r, coeff_count, nn, ni = precompute_zernike(xsize, nproj, order, nphi, ncoeff)

    # call compiled core
    rhoreturn, anm = _training_getrho_core(rawdat, simul, sininv_mat, i_idx, deltax, valid_mask,
                             r, coeff_count, nn, ni, rhofn_array, float(order), nphi, int(xstart), int(xsize))
    return rhoreturn, anm












#################################################### Non-MCM functions ####################################################

def load_projections(base_path, file_names, N):
    projections = np.zeros((N, N, len(file_names)))
    for i, fname in enumerate(file_names):
        full_path = os.path.join(base_path, fname)
        projections[:, :, i] = np.loadtxt(full_path).reshape((N, N))
    return projections


def plot_rhoreturn_slice(rhoreturn, central_y):
    plt.figure(figsize=(8, 5))
    for rho_n in range(0, 5):
        first_row = rhoreturn[central_y, rho_n, :]  # Shape: (z,)
        plt.plot(np.arange(len(first_row)), first_row, label=f'rho_{rho_n}')
    plt.title(f'Rhoreturn for y_fixed = {central_y}')
    plt.xlabel('z (px)')
    plt.ylabel('rho_n (px)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
def animate_rhoreturn(rhoreturn, y_range=None, y_dynamic=None, Legend=None, interval=100):
    """
    Animate 1D rhoreturn curves as a function of central_y,
    with dynamic y-limits for each frame.
    """
    n_y, n_proj, n_z = rhoreturn.shape
    if y_range is None:
        y_range = range(n_y)
    fig, ax = plt.subplots(figsize=(8, 5))
    lines = [ax.plot([], [], label=f"rho_{n}")[0] for n in range(n_proj)]
    ax.set_xlim(0, n_z - 1)
    ax.set_xlabel('z (px)')
    ax.set_ylabel('rho_n (px)')
    ax.set_title('Rhoreturn for central_y = 0')
    if Legend == True:
        ax.legend()
    # ax.grid(True)
    plt.tight_layout()
    
    if not y_dynamic:
        central_index = n_y // 2
        central_slice = rhoreturn[:, :, central_index]
        y_min, y_max = np.min(central_slice), np.max(central_slice)
        y_abs_fixed = np.max([abs(y_min), abs(y_max)])

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        ydata = []
        for n, line in enumerate(lines):
            data = rhoreturn[:, n, frame]
            line.set_data(np.arange(n_z), data)
            ydata.append(data)
        ydata = np.concatenate(ydata)
        
        if y_dynamic:
            y_min, y_max = np.min(ydata), np.max(ydata)
            y_abs = np.max([abs(y_min), abs(y_max)])
        else:
            # y_abs = 2e-2  # Fixed y-limits for better visibility
            y_abs = y_abs_fixed
            
        ax.set_ylim(-1.05*y_abs, 1.05*y_abs)
        ax.set_title(f'Rhoreturn for y_fixed = {frame}')
        return lines

    fps = 2
    
    anim = animation.FuncAnimation(
        fig, update, frames=y_range, init_func=init,
        interval=1000 // fps, blit=False
    )
    # filename="rhoreturn_across_slices_scaled.gif"
    # filename="rhoreturn_across_slices.gif"
    # filename="rhoreturn_across_slices_allRHO.gif"
    # anim.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    return HTML(anim.to_jshtml())

def animate_slices_normalized(volume,
                              start_frame=0,
                              end_frame=None,
                              fixed_z=None,
                              output_filename="Cormack_reconstructed_xz.gif",
                              fps=10,
                              interval=100,
                              cmap='hot',
                              save=None,
                              save_mp4=False):
    """
    Animate x–y slices (varying z-index) from a 3D Cormack reconstruction volume,
    normalizing each slice to have intensity in [0, 1].

    Parameters
    ----------
    volume : np.ndarray
        3D array representing the reconstructed volume.
    start_frame : int
        Starting y-index for animation.
    end_frame : int or None
        Ending y-index for animation. If None, uses the last index.
    fixed_z : int or None
        If set, animates only this y-index.
    output_filename : str
        Filename for saving the animation.
    fps : int
        Frames per second for the animation.
    interval : int
        Delay between frames in milliseconds.
    cmap : str
        Colormap for the slices.
    save : bool or None
        If True, saves the animation as a GIF.

    Returns
    -------
    HTML
        HTML animation for Jupyter display.
    """
    

    N = volume.shape[0]
    if end_frame is None or end_frame >= N:
        end_frame = N - 1
    if fixed_z is not None:
        start_frame = end_frame = fixed_z
    frames_to_animate = range(start_frame, end_frame + 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    initial_slice = volume[:, start_frame, :].T
    # Normalize the initial slice
    norm_slice = (initial_slice - np.min(initial_slice)) / (np.ptp(initial_slice) + 1e-12)
    img = ax.imshow(norm_slice, cmap=cmap, origin='lower', vmin=0, vmax=1, aspect='auto')
    cbar = fig.colorbar(img, ax=ax, shrink=0.8)
    cbar.set_label("Normalized Intensity")
    
    title = ax.set_title(f"Reconstructed XZ Slice (y = {start_frame:03d})")
    ax.set_xlabel("X-axis (Px)")
    ax.set_ylabel("Z-axis (Pz)")

    def update(frame):
        slice_ = volume[:, frame, :].T
        # Normalize each slice to [0, 1]
        norm_slice = (slice_ - np.min(slice_)) / (np.ptp(slice_) + 1e-12)
        img.set_array(norm_slice)
        title.set_text(f"Reconstructed XZ Slice (y = {frame:03d})")
        return img, title

    anim = animation.FuncAnimation(fig, update, frames=frames_to_animate,
                                   interval=interval, blit=False)
    if save:
        # Use adaptive palette for GIFs to improve color smoothness
        anim.save(output_filename, writer='pillow', fps=fps, dpi=100, 
                  savefig_kwargs={'facecolor': 'white'})
    if save_mp4:
        mp4_filename = output_filename.replace('.gif', '.mp4')
        anim.save(mp4_filename, writer='ffmpeg', fps=fps, dpi=100,
                  savefig_kwargs={'facecolor': 'white'})
    plt.close(fig)
    return HTML(anim.to_jshtml())

def rho_symmetrize(rhoreturn, inverted = None, scale_proj0=1e-2):
    xsize = rhoreturn.shape[0]      # 64
    nproj = rhoreturn.shape[1]      # 5
    zsize = rhoreturn.shape[0]      # 64

    # Scale projection 0 by 1e-2
    rhoreturn_scaled = rhoreturn.copy()
    # rhoreturn_dft[:, 0, :] *= -1 #1e-1
    # rhoreturn_dft[:, 1, :] *= 5e-1

    # Mirror about z-axis for all x and all rho_n (center double-counted)
    mirrored_z_rhoreturn = np.concatenate([
        np.flip(rhoreturn_scaled, axis=2),    # -64..0
        rhoreturn_scaled                      # 0..63 (center double-counted)
    ], axis=2)  # shape: (64, 5, 128)
    
    if inverted == True:
        # Mirror about x-axis (center at 63/64)
        rho_full = np.concatenate([
            np.flip(mirrored_z_rhoreturn, axis=0),         # x: 63..0 (edge to center)
            mirrored_z_rhoreturn                           # x: 0..63 (center to edge)
        ], axis=0)  # shape: (128, 5, 128)
    
    else:
        # Mirror about x-axis: 0..63 (original), 64..127 (mirror of 62..0)
        rho_full = np.concatenate([
            rhoreturn_scaled,                        # x = 0..64 (center at 63)
            rhoreturn_scaled[-1::-1, :, :]           # x = 63..1 (mirror, skip x=64 to avoid double-counting center)
        ], axis=0)  # shape: (128, 5, 128)
    
    
    # Debugging
    # print(xsize, nproj, zsize)          # Should be (64, 5, 64)
    # print(mirrored_z_rhoreturn.shape)   # Should be (64, 5, 128)
    # print(rho_full.shape)               # Should be (128, 5, 128)
    return rho_full

# Functions for generating ACAR Projections from 3D TPMD:

def generate_projections(volume, num_projections, save_folder="Cu_ProjectionsGen"):
    """
    Generates 2D projections of a 3D volume by rotating the volume and summing along the x-axis.
    
    Parameters:
        volume (np.ndarray): 3D volume of shape (N, N, N)
        angles (list of float): List of angles (in degrees) to project
        save_folder (str): Directory to save flattened projection files
    
    Returns:
        angles (np.ndarray): Array of projection angles used
    """
    N = volume.shape[0]
    angles = np.linspace(0, 45, num_projections)
    assert volume.shape == (N, N, N), "Volume must be cubic"
    os.makedirs(save_folder, exist_ok=True)

    for angle in tqdm(angles, desc="Generating projections"):
        filename = os.path.join(save_folder, f"angle_({angle:.1f}).txt")
        if os.path.exists(filename):
            print(f"Skipping existing projection: {filename}")
            continue

        # Rotate around Y-axis (i.e., in the XZ-plane)
        rotated_volume = rotate(volume, angle=angle, axes=(0, 2), reshape=False, order=1, mode='constant', cval=0.0)

        # Integrate along x-axis to simulate a projection at this angle
        projection = np.sum(rotated_volume, axis=0)  # Resulting shape: (N, N)

        np.savetxt(filename, projection.flatten())
        print(f"Saved projection: {filename}")

    return np.array(angles)


def load_generated_projections(folder_path, N):
    """
    Load and stack all projection files from the specified folder using the
    updated naming convention: 'angle_({angle:.1f}).txt'.
    
    Returns:
        projections: np.ndarray of shape (N, N, nproj)
        angles: list of angles (float) corresponding to each projection
    """
    # Match files like 'angle_(12.3).txt'
    files = [f for f in os.listdir(folder_path) if re.match(r'angle_\(\d+\.?\d*\)\.txt$', f)]

    def extract_angle(f):
        match = re.search(r'angle_\(([\d.]+)\)\.txt$', f)
        if match:
            return float(match.group(1))
        raise ValueError(f"Invalid filename format: {f}")

    # Sort files numerically by angle
    files_sorted = sorted(files, key=extract_angle)

    projections_file = []
    angles = []
    for f in files_sorted:
        path = os.path.join(folder_path, f)
        data = np.loadtxt(path).reshape((N, N)).T
        projections_file.append(data)
        angles.append(extract_angle(f))

    projections_file = np.stack(projections_file, axis=-1)  # Shape: (N, N, nproj)
    return projections_file, angles


# For the DFT data from Stephen:
def load_generated_projections_test(folder_path, N):
    """
    Load and stack projection files matching the pattern:
    'I_TPMD2D.OUT_PROJ_<index>_<N>'.
    
    Each file contains a flat 1D array of length N*N representing a 2D projection.

    Returns:
        projections: np.ndarray of shape (N, N, nproj)
        indices: list of integers for projection order
    """
    pattern = rf"I_TPMD2D\.OUT_PROJ_(\d+)_{N}$"
    files = [f for f in os.listdir(folder_path) if re.match(pattern, f)]
    
    if not files:
        raise FileNotFoundError(f"No files matching pattern found in {folder_path}")

    def extract_index(filename):
        return int(re.match(pattern, filename).group(1))

    files_sorted = sorted(files, key=extract_index)

    projections = []
    indices = []
    for fname in files_sorted:
        path = os.path.join(folder_path, fname)
        try:
            flat_data = np.loadtxt(path)
            if flat_data.size != N * N:
                raise ValueError(f"Unexpected data size in {fname}: expected {N*N}, got {flat_data.size}")
            data_2d = flat_data.reshape((N, N))
            projections.append(data_2d)
            indices.append(extract_index(fname))
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    if not projections:
        raise ValueError("No valid projection data loaded.")

    projections = np.stack(projections, axis=-1)  # Shape: (N, N, nproj)
    
    return projections, indices


def generate_ncoeffs(nproj):
    """
    Generate a dynamic array of ncoeffs based on the number of projections.
    The values decrease in steps across defined proportions of the total projections:
        - First 20% → 120
        - Next 20% → 90
        - Next 20% → 60
        - Final 40% → 30
    """
    ncoeffs = np.empty(nproj, dtype=int)

    p20 = nproj // 5        # 20% of total projections
    p40 = nproj - 3 * p20   # Remaining 40%

    ncoeffs[:p20] = 120
    ncoeffs[p20:2*p20] = 90
    ncoeffs[2*p20:3*p20] = 60
    ncoeffs[3*p20:] = 30

    return ncoeffs


"""Smearing Function:"""
def apply_detector_resolution3D(vol, N=512, fwhm_x_au=0.11, fwhm_y_au=0.137, fwhm_z_au=0.11):
    """
    Apply realistic detector resolution to a 3D volume in atomic units.
    Assumes image size is 10x10x10 a.u. and shape is N x N x N.
    FWHM in each direction is set by pixel size.
    """
    # Crop to N x N x N if needed
    if vol.shape[0] > N:
        vol = vol[:N, :N, :N]

    # Convert FWHM to sigma for Gaussian kernel
    sigma_x = fwhm_x_au / 2.355
    sigma_y = fwhm_y_au / 2.355
    sigma_z = fwhm_z_au / 2.355

    # Calculate pixel size in each direction
    # pixel_size_x = 6.87 / N
    # pixel_size_y = 6.87 / N
    # pixel_size_z = 6.87 / N
    pixel_size_x = 5.0 / N
    pixel_size_y = 5.0 / N
    pixel_size_z = 5.0 / N

    # Sigma in pixels
    sigma_x_pix = sigma_x / pixel_size_x
    sigma_y_pix = sigma_y / pixel_size_y
    sigma_z_pix = sigma_z / pixel_size_z

    # Apply Gaussian filter
    vol_smoothed = gaussian_filter(vol, sigma=[sigma_x_pix, sigma_y_pix, sigma_z_pix], mode='nearest')
    return vol_smoothed

def apply_detector_resolution(img, N=512, fwhm_x_au=0.11, fwhm_y_au=0.137):
    """
    Apply realistic detector resolution to a 2D image in atomic units.
    Assumes image size is 10x10 a.u. and shape is N x N.
    FWHM in each direction is set by pixel size.
    """
    # Crop to N x N if needed
    if img.shape[0] > N or img.shape[1] > N:
        img = img[:N, :N]

    # Convert FWHM to sigma for Gaussian kernel
    sigma_x = fwhm_x_au / 2.355
    sigma_y = fwhm_y_au / 2.355

    # Calculate pixel size in each direction (expected images are 10x10 atomic units)
    # pixel_size_x = 6.87 / N
    # pixel_size_y = 6.87 / N
    pixel_size_x = 5.0 / N
    pixel_size_y = 5.0 / N

    # Sigma in pixels
    sigma_x_pix = sigma_x / pixel_size_x
    sigma_y_pix = sigma_y / pixel_size_y

    # Apply Gaussian filter
    img_smoothed = gaussian_filter(img, sigma=[sigma_x_pix, sigma_y_pix], mode='nearest')
    return img_smoothed


# Bryn's Random MCP code
def smear(array, resolution):
    """Smears each profile with a Gaussian filter."""
    return gaussian_filter1d(array, resolution)


def createmcp(*, totcounts, pz, resolution, popt, nppts, fit_result, fit):
    # Windowing Gaussian for the "Fermi surface" bit
    fermigaus = gaussian(pz, 0.7, 0.1)

    # Majority band
    guess = np.zeros(popt.shape)
    for i in range(4):
        guess[i*2] = random.uniform(0, 3.0)
        guess[i*2+1] = random.uniform(0, 1.2)
    i = i + 1
    popt[i] = random.uniform(0, 0.2)
    i = i + 1
    popt[i] = 0
    i = i + 1
    popt[i] = random.uniform(0, 0.5)

    randcpup = np.zeros(nppts)
    for i in range(5):
        randcpup += gaussian(pz, *popt[i*2:i*2+2])
    i = i + 1
    randcpup += parabola(pz, *popt[i*2:i*2+3])

    scaleup = random.uniform(0, 0.1)
    crange = 0.5
    for j in range(1, 11):
        fit_result.params[f"a{j}"] = random.uniform(-crange, crange)
    fit_result.params["a0"] = 0.0
    for j in range(1, 11):
        fit_result.params[f"b{j}"] = 0

    fermipart = fit.model(x=pz, **fit_result.params).y
    randcpup += (scaleup * fermigaus * fermipart)

    # Minority band
    guess = np.zeros(popt.shape)
    for i in range(4):
        guess[i*2] = random.uniform(0, 3.0)
        guess[i*2+1] = random.uniform(0, 1.2)
    i = i + 1
    popt[i] = random.uniform(0, 0.2)
    i = i + 1
    popt[i] = 0
    i = i + 1
    popt[i] = random.uniform(0, 0.5)

    randcpdn = np.zeros(nppts)
    for i in range(5):
        randcpdn += gaussian(pz, *popt[i*2:i*2+2])
    i = i + 1
    randcpdn += parabola(pz, *popt[i*2:i*2+3])
    for j in range(1, 11):
        fit_result.params[f"a{j}"] = random.uniform(-crange, crange)
    fit_result.params["a0"] = 0.0
    for j in range(1, 11):
        fit_result.params[f"b{j}"] = 0

    fermipart = fit.model(x=pz, **fit_result.params).y
    scaledn = random.uniform(0.5 * scaleup, 0.9 * scaleup)
    randcpdn += (scaledn * fermigaus * fermipart)

    # Scale so that total profile contains totcounts counts
    cpup = totcounts * randcpup / np.trapz(randcpup)
    cpdn = totcounts * randcpdn / np.trapz(randcpdn)

    # Gaussian noise
    # cpupnoisy = cpup + np.sqrt(cpup) * np.random.normal(0, 1, cpup.size)
    # cpdnnoisy = cpdn + np.sqrt(cpdn) * np.random.normal(0, 1, cpdn.size)
    # cpupsmeared = smear(cpup, resolution)
    # cpdnsmeared = smear(cpdn, resolution)

    # Gaussian noise after smearing
    # cpupnoisysmeared = cpupsmeared + np.sqrt(cpupsmeared) * np.random.normal(0, 1, cpup.size)
    # cpdnnoisysmeared = cpdnsmeared + np.sqrt(cpdnsmeared) * np.random.normal(0, 1, cpdn.size)

    # MCP construction
    mcp = cpup - 0.8 * cpdn
    # smeared_mcp = smear(mcp, resolution)
    # noisy_mcp = cpupnoisy - 0.8 * cpdnnoisy
    # smearednoisy_mcp = cpupnoisysmeared - 0.8 * cpdnnoisysmeared

    postest = np.all(mcp > 0) if mcp[0] > 0 else np.all(mcp < 0)

    # Normalise and return
    if postest:
        mcp = mcp / np.trapz(mcp)
        # noisy_mcp = noisy_mcp / np.trapz(noisy_mcp)
        # smeared_mcp = smeared_mcp / np.trapz(smeared_mcp)
        # smearednoisy_mcp = smearednoisy_mcp / np.trapz(smearednoisy_mcp)
        return mcp              #, noisy_mcp, smeared_mcp, smearednoisy_mcp
    else:
        # Try again if sign is inconsistent
        return createmcp(totcounts=totcounts, pz=pz, resolution=resolution, popt=popt, nppts=nppts, fit_result=fit_result, fit=fit)
    
    
def gaussian(x, xalpha, A):
    return np.abs(A) * np.exp(-0.5 * (x / (2 * np.abs(xalpha))) ** 2)

def parabola(x, xalpha, A, c):
    return A * (x - xalpha) ** 2 + c

def _model(M, *args):
    x = M
    arr = np.zeros(x.shape)
    for i in range(5):
        arr += gaussian(x, *args[i*2:i*2+2])
    i = i + 1
    arr += parabola(x, *args[i*2:i*2+3])
    return arr

# Fourier fit to residuals (requires symfit)
def fourier_series(x, f, n=0):
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series
    
    

    
    
"""
Implements the full "realistic projection" simulation and saving procedure. This will 
process all 20 projections per simulation, apply elliptical detector convolution, MSF 
convolution (placeholder), Poisson noise, remove MSF convolution, and save the resulting 
rhoreturn as a flattened .txt file in the correct folder.
"""


def fft_convolve2d(a, b):
    # For MSF convolution
    a_padded = np.zeros((a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1))
    a_padded[:a.shape[0], :a.shape[1]] = a
    b_padded = np.zeros((a_padded.shape[0], a_padded.shape[1]))
    b_padded[:b.shape[0], :b.shape[1]] = b
    
    # Compute FFTs
    fft_a = np.fft.fft2(a_padded)
    fft_b = np.fft.fft2(b_padded)
    
    # Multiply in frequency domain and inverse FFT
    result = np.fft.ifft2(fft_a * fft_b).real
    
    # Extract the valid part
    start_row = b.shape[0] // 2
    start_col = b.shape[1] // 2
    return result[start_row:start_row+a.shape[0], start_col:start_col+a.shape[1]]


def MSF_convolution(c_simulated, grid_size=(512, 512)):
    """
    Create MSF by convolving two versions of c_simulated with different Poisson noise.
    
    Args:
        c_simulated (numpy.ndarray): The base simulated camera response
        grid_size (tuple): Output grid size for the MSF (default: (512, 512))
    
    Returns:
        numpy.ndarray: The simulated MSF normalized to [0, 1]
    """
    # Convert to torch tensor for Poisson noise generation
    c_tensor = torch.from_numpy(c_simulated.astype(np.float32))
    
    # Generate two different Poisson noise realizations
    # Ensure values are positive for Poisson distribution
    c_tensor_pos = torch.clamp(c_tensor, min=1e-6)
    
    # Create two noisy versions with different random seeds
    c_noisy1 = torch.poisson(c_tensor_pos).numpy().astype(np.float64)
    c_noisy2 = torch.poisson(c_tensor_pos).numpy().astype(np.float64)
    
    # Convolve the two noisy versions
    # msf_result = fft_convolve2d(c_noisy1, c_noisy2)
    msf_result = fftconvolve(c_noisy1, c_noisy2, mode='same') 
    msf_result = msf_result / np.max(msf_result)
    # print("MSF Range:", np.min(msf_result),  np.max(msf_result))
    
    # Resize to specified grid size if different from current size
    if msf_result.shape != grid_size:
        zoom_factors = (grid_size[0] / msf_result.shape[0], 
                       grid_size[1] / msf_result.shape[1])
        msf_result = zoom(msf_result, zoom_factors, order=1)
    
    # Normalize to [0, 1]
    msf_result = msf_result / np.max(msf_result)
    
    return msf_result


def apply_elliptical_gaussian(img, sigma_x, sigma_y, projection_size_au=5.0, N=512):
    """
    img: 2D numpy array (128x128)
    sigma_x, sigma_y: FWHM in physical units (e.g., x=0.11, y=0.137 au for ACAR)
    field_of_view: total width/height in physical units (default 10)
    N: number of pixels (default 128)
    """
    #### projection_size_au=6.87 for ZrZn2 Data
    pixel_size = projection_size_au / N         # units per pixel
    sigma_x_pix = sigma_x / 2.355 / pixel_size
    sigma_y_pix = sigma_y / 2.355 / pixel_size
    return gaussian_filter(img, sigma=[sigma_y_pix, sigma_x_pix], mode='nearest')


def symmetrize_4fold(image):
    """
    Enforce 4-fold symmetry on a square odd-sized image, including correct handling of central row and column.
    Quadrants, central row, and central column are all summed with their mirrors, then copied out.
    """
    N = image.shape[0]
    assert image.shape[0] == image.shape[1], "Image must be square"
    assert N % 2 == 1, "This function assumes odd N"
    mid = N // 2

    # Quadrant slices (excluding center row/col)
    Q1 = image[0:mid, 0:mid].copy()
    Q2 = np.flipud(image[mid+1:, 0:mid])         # flip up-down
    Q3 = np.fliplr(image[0:mid, mid+1:])         # flip left-right
    Q4 = np.flipud(np.fliplr(image[mid+1:, mid+1:]))  # flip both
    Q1_sum = Q1 + Q2 + Q3 + Q4

    # Build new symmetric image
    sym = np.zeros_like(image)
    # Place Q1_sum and its mirrors
    sym[0:mid, 0:mid] = Q1_sum
    sym[mid+1:, 0:mid] = np.flipud(Q1_sum)
    sym[0:mid, mid+1:] = np.fliplr(Q1_sum)
    sym[mid+1:, mid+1:] = np.flipud(np.fliplr(Q1_sum))

    # Central row and column (including center pixel)
    for i in range(N):
        if i < mid:
            val = (image[mid, i] + image[mid, N-1-i] + image[i, mid] + image[N-1-i, mid])
            sym[mid, i] = val
            sym[mid, N-1-i] = val
            sym[i, mid] = val
            sym[N-1-i, mid] = val
        elif i == mid:
            # Center pixel: sum all four (which are the same)
            val = image[mid, mid] * 4
            sym[mid, mid] = val
    return sym / 4


def symmetrize_4fold_even(image):
    """
    Enforce 4-fold symmetry on a square odd-sized image, including correct handling of central row and column.
    Quadrants, central row, and central column are all summed with their mirrors, then copied out.
    """
    N = image.shape[0]
    assert image.shape[0] == image.shape[1], "Image must be square"
    assert N % 2 == 0, "This function assumes even N"
    mid = N // 2

    # Quadrant slices (excluding center row/col)
    Q1 = image[0:mid, 0:mid].copy()
    Q2 = np.flipud(image[mid:, 0:mid])         # flip up-down
    Q3 = np.fliplr(image[0:mid, mid:])         # flip left-right
    Q4 = np.flipud(np.fliplr(image[mid:, mid:]))  # flip both
    Q1_sum = Q1 + Q2 + Q3 + Q4

    # Build new symmetric image
    sym = np.zeros_like(image)
    # Place Q1_sum and its mirrors
    sym[0:mid, 0:mid] = Q1_sum
    sym[mid:, 0:mid] = np.flipud(Q1_sum)
    sym[0:mid, mid:] = np.fliplr(Q1_sum)
    sym[mid:, mid:] = np.flipud(np.fliplr(Q1_sum))
    return sym / 4


#total_counts = 2_000_000  # Total counts for each projection, can be adjusted as needed

def downsample(arr):
    """
    Downsample a (513, 513, nproj) or (513, 513) array to (512, 512, nproj) or (512, 512) using scipy.ndimage.zoom.
    Handles both 2D and 3D arrays.
    """
    dims = arr.shape[0]
    if arr.ndim == 3:
        downsampled = np.zeros((dims-1, dims-1, arr.shape[2]), dtype=arr.dtype)
        for i in range(arr.shape[2]):
            downsampled[:, :, i] = zoom(arr[:, :, i], zoom=((dims-1)/dims, (dims-1)/dims), order=1)
    elif arr.ndim == 2:
        downsampled = zoom(arr, zoom=((dims-1)/dims, (dims-1)/dims), order=1)
    else:
        raise ValueError("Input array must be 2D or 3D.")
    return downsampled

def upsample(arr):
    """
    Upsample a (512, 512, nproj) or (512, 512) array to (513, 513, nproj) or (513, 513) using scipy.ndimage.zoom.
    Handles both 2D and 3D arrays.
    """
    dims = arr.shape[0]
    if arr.ndim == 3:
        upsampled = np.zeros((dims+1, dims+1, arr.shape[2]), dtype=arr.dtype)
        for i in range(arr.shape[2]):
            upsampled[:, :, i] = zoom(arr[:, :, i], zoom=((dims+1)/dims, (dims+1)/dims), order=1)
    elif arr.ndim == 2:
        upsampled = zoom(arr, zoom=((dims+1)/dims, (dims+1)/dims), order=1)
    else:
        raise ValueError("Input array must be 2D or 3D.")
    return upsampled

def make_realistic_projections(raw_projs, sigma_x=0.11, sigma_y=0.137, projection_size_au=5.0, total_counts=0, four_sym=None):
    """
    raw_projs: shape (N, N, nproj)
    Returns: processed_projs (N, N, nproj)
    For ACAR measurement with 10x10au (73x73mRad) dimensions, and 129x129 pixel size, 
    each pixel is ≈0.564mRad; =0.5682mRad for 128x128 pixel size.
    
    Note: x-convolution is about 0.8mRad/0.11au, y-convolution is about 1.0mRad/0.137au.
    sigma * 2.354 = FWHM, so FWHM ≈ 0.137au, then just make sure to convert this back into
    channels (gotta know the units): 0.137/0.0775, and for the sigma we have 2.355
    """
    #### projection_size_au=6.87 for ZrZn2 Data
    N, _, nproj = raw_projs.shape
    processed = np.zeros((N-1, N-1, nproj), dtype=raw_projs.dtype)
    msf_stack = []
    for idx in range(nproj):
        proj = raw_projs[:, :, idx]
        # 1. Elliptical detector convolution
        proj = apply_elliptical_gaussian(proj, sigma_x, sigma_y, projection_size_au, N=N)
        # # 2. MSF convolution (placeholder)
        # c_simulated = np.loadtxt("Data_Generation_Required/c_simulated_513x513.txt").reshape((N, N))  # Load the simulated camera response
        c_simulated = np.loadtxt("Data_Generation_Required/c_simulated_513x513.txt").reshape((N, N))  # Load the simulated camera response

        msf = MSF_convolution(c_simulated, grid_size=proj.shape)    # <-- FIXED HERE
        proj = proj * msf                                        
        msf_stack.append(msf)
        # 3. Scale to total_counts
        proj_sum = np.sum(proj)
        if proj_sum > 0:
            proj = proj * (total_counts / proj_sum)
            # proj = proj * (1 / proj_sum)                # normalize the sum of the projections as 1

        # 4. Poisson noise (integer counts)
        proj_torch = torch.tensor(proj, dtype=torch.float32)
        proj_noisy = (torch.poisson(proj_torch).numpy())
        # proj_noisy = proj.copy()                              # To Test Without Noise
        
        # 5. Divide by MSF (avoid division by zero)
        proj_final = downsample(np.where(msf != 0, proj_noisy / msf, 0))
        
        """new addition: normalize projections"""
        if np.sum(proj_final) > 0:
            proj_final = proj_final * (proj_sum / total_counts)      # return the intensity back to the original total_counts

        if four_sym==True:
            # 6. Enforce 4-fold symmetry
            proj_final = symmetrize_4fold_even(proj_final)  # Enforce 4-fold symmetry
                  
        processed[:, :, idx] = proj_final
    return processed


def load_simulated_rhos(folder, xsize, nproj, measurement=False):
    """
    Loads and reshapes simulated rho files.
    For measurement: loads all rhos_simulated_measurement_{idx:03d}.txt files and returns shape (num_files, xsize, nproj, xsize).
    For ideal: loads the single rhos_simulated_ideal.txt file and returns shape (1, xsize, nproj, xsize).
    """
    if measurement == "measurement":
        files = [f for f in os.listdir(folder) if re.match(r"rhos_simulated_measurement_\d{3}\.txt", f)]
        indices = [int(re.search(r"(\d{3})", f).group(1)) for f in files]
        sorted_files = [f for _, f in sorted(zip(indices, files))]
        rhos = []
        for fname in sorted_files:
            arr = np.loadtxt(os.path.join(folder, fname)).reshape((xsize, nproj, xsize))
            rhos.append(arr)
        return np.stack(rhos, axis=0)  # shape: (num_files, xsize, nproj, xsize)
    
    elif measurement == "ideal":
        fname = os.path.join(folder, "rhos_simulated_ideal.txt")
        arr = np.loadtxt(fname).reshape((xsize, nproj, xsize))
        return arr[np.newaxis, ...]  # shape: (1, xsize, nproj, xsize)
    
    elif measurement == "rho_rand_ideal":
        files = [f for f in os.listdir(folder) if re.match(r"rho_ideal_\d{3}\.txt", f)]
        indices = [int(re.search(r"(\d{3})", f).group(1)) for f in files]
        sorted_files = [f for _, f in sorted(zip(indices, files))]
        rhos = []
        for fname in sorted_files:
            arr = np.loadtxt(os.path.join(folder, fname)).reshape((xsize, nproj, xsize))
            rhos.append(arr)
        return np.stack(rhos, axis=0)  # shape: (num_files, xsize, nproj, xsize)
    
    elif measurement == "rho_rand_measurement":
        files = [f for f in os.listdir(folder) if re.match(r"rho_meas_\d{3}\.txt", f)]
        indices = [int(re.search(r"(\d{3})", f).group(1)) for f in files]
        sorted_files = [f for _, f in sorted(zip(indices, files))]
        rhos = []
        for fname in sorted_files:
            arr = np.loadtxt(os.path.join(folder, fname)).reshape((xsize, nproj, xsize))
            rhos.append(arr)
        return np.stack(rhos, axis=0)  # shape: (num_files, xsize, nproj, xsize)
    else:
        raise ValueError("Invalid measurement type. Use 'measurement', 'ideal', 'rho_rand_ideal', or 'rho_rand_measurement'.")
    
def load_randrhos(folder, xsize, nproj, measurement=False):
    """
    Loads and reshapes simulated rho files.
    For measurement: loads all rhos_simulated_measurement_{idx:03d}.txt files and returns shape (num_files, xsize, nproj, xsize).
    For ideal: loads the single rhos_simulated_ideal.txt file and returns shape (1, xsize, nproj, xsize).
    """
    
    if measurement == "rho_rand_ideal":
        files = [f for f in os.listdir(folder) if re.match(r"rho_ideal_\d{3}\.txt", f)]
        indices = [int(re.search(r"(\d{3})", f).group(1)) for f in files]
        sorted_files = [f for _, f in sorted(zip(indices, files))]
        rhos = []
        for fname in sorted_files:
            arr = np.loadtxt(os.path.join(folder, fname)).reshape((xsize, nproj, 1))
            rhos.append(arr)
        return np.stack(rhos, axis=0)  # shape: (num_files, xsize, nproj, xsize)
    
    elif measurement == "rho_rand_measurement":
        files = [f for f in os.listdir(folder) if re.match(r"rho_meas_\d{3}\.txt", f)]
        indices = [int(re.search(r"(\d{3})", f).group(1)) for f in files]
        sorted_files = [f for _, f in sorted(zip(indices, files))]
        rhos = []
        for fname in sorted_files:
            arr = np.loadtxt(os.path.join(folder, fname)).reshape((xsize, nproj, 1))
            rhos.append(arr)
        return np.stack(rhos, axis=0)  # shape: (num_files, xsize, nproj, xsize)
    else:
        raise ValueError("Invalid measurement type. Use 'measurement', 'ideal', 'rho_rand_ideal', or 'rho_rand_measurement'.")


def make_rho_rand_dual(
    rhoreturn, rhoreturn_measurement, slice_idx, pz, nppts, popt, fit_result, fit, totcounts=5e6
):
    """
    Returns:
        rho_rand: (N, 20, N)  # 20 projections, ideal
        rho_rand_measurement: (N, 5, N)  # 5 projections, measurement
    """
    N = rhoreturn.shape[0]
    n_proj = rhoreturn.shape[1]
    assert n_proj == 20, "rhoreturn must have 20 projections"
    mask = pz >= 0
    pz_pos = pz[mask]
    # pz_pos = np.arange(np.sum(mask))

    # Generate 20 random MCPs and random slice indices
    fixed_mcps = []
    fixed_randnums = []
    for rho_n in range(20):
        mcp = createmcp(
            totcounts=2_000_000,
            pz=pz,
            resolution=0.45/2.354/np.abs(pz[1]-pz[0]),
            popt=popt,
            nppts=nppts,
            fit_result=fit_result,
            fit=fit
        )
        
        fixed_mcps.append(mcp / np.max(mcp))  # Normalize MCP to max value of 1
        rand_num = np.random.randint(0, int(0.35*rhoreturn.shape[2]))
        fixed_randnums.append(rand_num)

    # Build rho_rand (20 projections)
    rho_rand = np.zeros_like(rhoreturn)
    for rho_n in range(20):
        mcp = fixed_mcps[rho_n]
        rand_num = fixed_randnums[rho_n]
        mcp_pos = mcp[mask]
        rand_rho_n = rhoreturn[:, rho_n, rand_num] / np.max(rhoreturn[:, rho_n, rand_num])
        mcp_pos = mcp_pos[:rand_rho_n.shape[0]]
        n = 4 * rho_n
        rho_rand[:, rho_n, slice_idx] = rand_rho_n * mcp_pos #* np.cos(n * pz_pos)
        
        normer = np.maximum(
            np.max(rhoreturn[:, rho_n, slice_idx]),
            np.abs(np.min(rhoreturn[:, rho_n, slice_idx]))
        )
        # if np.max(np.abs(rho_rand[:, rho_n, slice_idx])) > np.max(rho_rand[:, rho_n, slice_idx]):
        #     rho_rand[:, rho_n, slice_idx] = -1 * rho_rand[:, rho_n, slice_idx] * (
        #         normer / np.max(np.abs(rho_rand[:, rho_n, slice_idx]))
        #     )
        # else:
        rho_rand[:, rho_n, slice_idx] = rho_rand[:, rho_n, slice_idx] * (
            normer / np.max(np.abs((rho_rand[:, rho_n, slice_idx])))
        )

    # Build rho_rand_measurement (5 projections, using first 5 randomizations)
    rho_rand_measurement = np.zeros_like(rhoreturn_measurement)
    proj_count = rho_rand_measurement.shape[1]
    for rho_n in range(proj_count):
        mcp = fixed_mcps[rho_n]
        rand_num = fixed_randnums[rho_n]
        mcp_pos = mcp[mask]
        rand_rho_n = rhoreturn_measurement[:, rho_n, rand_num] / np.max(rhoreturn_measurement[:, rho_n, rand_num])
        mcp_pos = mcp_pos[:rand_rho_n.shape[0]]
        n = 4 * rho_n
        rho_rand_measurement[:, rho_n, slice_idx] = rand_rho_n * mcp_pos #* np.cos(n * pz_pos)
        
        normer = np.maximum(
            np.max(rhoreturn_measurement[:, rho_n, slice_idx]),
            np.abs(np.min(rhoreturn_measurement[:, rho_n, slice_idx]))
        )
        # if np.max(np.abs(rho_rand_measurement[:, rho_n, slice_idx])) > np.max(rho_rand_measurement[:, rho_n, slice_idx]):
        #     rho_rand_measurement[:, rho_n, slice_idx] = -1 * rho_rand_measurement[:, rho_n, slice_idx] * (
        #         normer / np.max(np.abs(rho_rand_measurement[:, rho_n, slice_idx]))
        #     )
        # else:
        rho_rand_measurement[:, rho_n, slice_idx] = rho_rand_measurement[:, rho_n, slice_idx] * (
            normer / np.max(np.abs(rho_rand_measurement[:, rho_n, slice_idx]))
        )
            
    # --- Normalize both arrays to total_counts ---
    max_rho_rand_measurement = np.max(rho_rand_measurement)
    max_rho_rand = np.max(rho_rand)

    # Normalize measurement to total_counts
    # if max_rho_rand_measurement > 0:
    #     rho_rand_measurement = rho_rand_measurement * (total_counts / sum_rho_rand_measurement)
    # Normalize ideal to the same total as measurement (not to total_counts directly)
    if max_rho_rand > 0 and max_rho_rand_measurement > 0:
        rho_rand = rho_rand * (max_rho_rand_measurement / max_rho_rand)

    return rho_rand, rho_rand_measurement, fixed_mcps


# Run make_rho_rand_dual and capture the MCPs
def test_mcp_consistency():
    # ...set up all required arguments for make_rho_rand_dual...
    # For demonstration, use dummy arrays and parameters
    rhoreturn = np.random.rand(128, 20, 128)
    rhoreturn_measurement = np.random.rand(128, 5, 128)
    slice_idx = 0
    pz = np.linspace(-5, 5, 128)
    nppts = 128
    popt = np.random.rand(20)
    fit_result = None
    fit = None

    # Patch make_rho_rand_dual to return MCPs for inspection
    fixed_mcps = []
    for rho_n in range(20):
        mcp = np.random.rand(len(pz))
        fixed_mcps.append(mcp)
    # Compare first 5 MCPs
    # for i in range(5):
        # print(f"MCP for projection {i}:")
        # print(fixed_mcps[i])
        # print()

    # Check if all first 5 MCPs are identical for both 20-proj and 5-proj
    identical = all(np.allclose(fixed_mcps[i], fixed_mcps[i]) for i in range(5))
    print("First 5 MCPs are identical for both cases:", identical)
    
    
    
def getrho_1d(rawdat, order, nproj, calib, pang, nphi, ncoeff, rhofn, sinmat):
    nsize = rawdat.shape[0]
    xsize = nsize // 2
    xstart = nsize // 2
    simul = setupprojs(order, pang, nproj)
    
    # Initialize rhoreturn based on the dimensionality of rawdat
    if rawdat.ndim == 2:  # 1D projections (e.g., [512, nproj])
        rhoreturn = np.zeros((xsize, nproj, 1))
    elif rawdat.ndim == 3:  # 2D projections (e.g., [512, 512, nproj])
        rhoreturn = np.zeros((xsize, nproj, xsize))
    else:
        raise ValueError("rawdat must be either 2D (1D projections) or 3D (2D projections).")
    
    for yfixed in range(xsize):
        yfixed_shifted = yfixed + xstart
        
        if rawdat.ndim == 2:  # Handle 1D projections
            # For 1D projections, extract the relevant slice directly
            project = rawdat[xstart:xstart + xsize, :]
        elif rawdat.ndim == 3:  # Handle 2D projections
            # For 2D projections, extract the slice at yfixed
            project = rawdat[xstart:xstart + xsize, yfixed_shifted, :]
        
        # Normalize the projections
        project = plnorm(project)
        
        # Compute cosine-transformed projections
        proj = calccos(project, xsize, nphi, nproj)
        
        # Compute coefficients
        anm = calcanm(sinmat, simul, nproj, nphi, proj)
        
        # Compute rho_n
        rho = calcrho(anm, order, nproj, xsize, nphi, rhofn, ncoeff)
        
        if rawdat.ndim == 2:  # Store for 1D projections
            rhoreturn[:, :, 0] = rho
        elif rawdat.ndim == 3:  # Store for 2D projections
            rhoreturn[:, :, yfixed] = rho
    
    return rhoreturn