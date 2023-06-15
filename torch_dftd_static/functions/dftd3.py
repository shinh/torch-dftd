"""pytorch implementation of Grimme's D3 method"""  # NOQA
from typing import Dict, Optional

import torch
from torch import Tensor
from torch_dftd.functions.smoothing import poly_smoothing

from torch_dftd.functions.dftd3 import d3_k1, d3_k3

def _mod(a: Tensor, b: Tensor):  # abs(_mod(a, b)) <= b/2
    return a - torch.floor(a / b + 0.5) * b

def _small_matmul(x: Tensor, mat: Tensor):  # compute x @ mat for small mat
    assert x.size(-1) == mat.size(0), f"wrong shape for matmul: {x.shape} * {mat.shape}"
    x_sliced = [s.squeeze(-1) for s in torch.split(x, 1, dim=-1)]
    y_sliced = [torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device) for _ in range(x.size(-1))]
    for k in range(mat.size(0)):
        for j in range(mat.size(1)):
            y_sliced[j] = y_sliced[j] + x_sliced[k] * mat[k, j]
    y = torch.stack(y_sliced, dim=-1)
    return y

def edisp(  # calculate edisp by all-pair computation
    Z: Tensor,
    pos: Tensor,  # (n_atoms, 3)
    shift_int: Tensor,
    needs_both_ij_ji: Tensor,
    cell: Tensor,
    cell_inv: Tensor,
    n_shifts: Tensor,
    c6ab: Tensor,
    r0ab: Tensor,
    rcov: Tensor,
    r2r4: Tensor,
    params: Dict[str, float],
    cutoff: Optional[float] = None,
    cnthr: Optional[float] = None,
    k1=d3_k1,
    k3=d3_k3,
    cutoff_smoothing: str = "none",
    damping: str = "zero",
    atom_mask: Optional[Tensor] = None,
    shift_mask: Optional[Tensor] = None,
):
    assert atom_mask is not None
    assert shift_mask is not None

    n_atoms = len(Z)
    triu_mask = (torch.arange(n_atoms, device=Z.device)[:, None] < torch.arange(n_atoms, device=Z.device)[None, :])[:, :, None] | needs_both_ij_ji[None, None, :]
    triu_mask = triu_mask & atom_mask[:, None, None] & atom_mask[None, :, None]
    triu_mask = triu_mask & shift_mask[None, None, :]

    # calculate pairwise distances
    cell_coordinate = _small_matmul(pos, cell_inv)
    shifted_cell_coordinate = cell_coordinate[:, None, :] + shift_int[None, :, :]
    v_atom_ghost_cell_coordinate = cell_coordinate[:, None, None, :] - shifted_cell_coordinate[None, :, :, :]
    v_atom_ghost_cell_coordinate = _mod(v_atom_ghost_cell_coordinate, n_shifts)
    v_atom_ghost = _small_matmul(v_atom_ghost_cell_coordinate, cell)
    r2 = torch.sum(v_atom_ghost ** 2, axis=-1)
    r = torch.sqrt(r2 + 1e-20)

    # calculate coordination numbers (n_atoms,)
    rco = rcov[Z][:, None] + rcov[Z][None, :]  # (n_atoms, n_atoms)
    rr = rco[:, :, None] / r  # (n_atoms, n_atoms, n_shift)
    damp = torch.sigmoid(k1 * (rr - 1.0))  # (n_atoms, n_atoms, n_shift)
    if cnthr is not None and cutoff_smoothing == "poly":
        damp *= poly_smoothing(r, cnthr)
    if cnthr is not None:
        damp = torch.where(r <= cnthr, damp, torch.tensor(0.0, device=damp.device))
    damp = torch.where(triu_mask, damp, torch.tensor(0.0, device=damp.device))
    damp = torch.sum(damp, axis=2)
    nc = torch.sum(damp, axis=1) + torch.sum(damp, axis=0)  # (n_atoms,)

    # calculate c6 and c8
    cn0 = c6ab[:, :, :, :, 0][Z, :][:, Z]
    cn1 = c6ab[Z, 1, :, 0, 1]  # (n_atoms, 5)
    cn2 = c6ab[1, Z, 0, :, 2]  # (n_atoms, 5)
    k3_rnc_1 = torch.where(cn1 >= 0.0, k3 * (nc[:, None] - cn1) ** 2, torch.tensor(-1.0e20, device=cn1.device))
    k3_rnc_2 = torch.where(cn2 >= 0.0, k3 * (nc[:, None] - cn2) ** 2, torch.tensor(-1.0e20, device=cn2.device))
    r_ratio_1 = torch.softmax(k3_rnc_1, dim=-1)
    r_ratio_2 = torch.softmax(k3_rnc_2, dim=-1)
    c6 = (cn0 * r_ratio_1[:, None, :, None] * r_ratio_2[None, :, None, :]).sum(dim=(-1,-2))
    c8c6_ratio = 3 * r2r4[Z][:, None] * r2r4[Z][None, :]
    c8 = c6 * c8c6_ratio

    # calculate energy
    s6 = params["s6"]
    s8 = params["s18"]
    r6 = r2 ** 3
    r8 = r6 * r2
    if damping in ["bj", "bjm"]:
        a1 = params["rs6"]
        a2 = params["rs18"]
        # Becke-Johnson damping, zero-damping introduces spurious repulsion
        # and is therefore not supported/implemented
        tmp = a1 * torch.sqrt(c8c6_ratio) + a2
        tmp2 = tmp ** 2
        tmp6 = tmp2 ** 3
        tmp8 = tmp6 * tmp2
        e6 = 1 / (r6 + tmp6[:, :, None])
        e8 = 1 / (r8 + tmp8[:, :, None])
    else:
        raise ValueError(f"[ERROR] Unexpected value damping={damping}")

    e6 = -0.5 * s6 * c6[:, :, None] * e6
    e8 = -0.5 * s8 * c8[:, :, None] * e8
    e68 = e6 + e8

    if cutoff is not None and cutoff_smoothing == "poly":
        e68 *= poly_smoothing(r, cutoff)

    e68 = torch.where(r <= cutoff, e68, torch.tensor(0.0, device=e68.device))

    e68 = torch.where(triu_mask, e68, torch.tensor(0.0, device=e68.device))

    return e68.to(torch.float64).sum() * 2.0
