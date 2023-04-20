"""pytorch implementation of Grimme's D3 method"""  # NOQA
from typing import Dict, Optional

import torch
from torch import Tensor
from torch_dftd.functions.smoothing import poly_smoothing

# conversion factors used in grimme d3 code

from torch_dftd.functions.dftd3 import d3_k1, d3_k3

def edisp(  # calculate edisp by all-pair computation
    Z: Tensor,
    pos: Tensor,  # (n_atoms, 3)
    shift_vecs: Tensor,  # half of shift vectors (all shift vecs = shift_vecs + -shift_vecs + [(0,0,0)])
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
    n_atoms = len(Z)
    #assert torch.all(shift_vecs[0] == 0.0)
    #triu_mask = (torch.arange(n_atoms)[:, None] < torch.arange(n_atoms)[None, :])[:, :, None] | ((torch.arange(len(shift_vecs)) > 0)[None, None, :])
    triu_mask = (torch.arange(n_atoms)[:, None] < torch.arange(n_atoms)[None, :])[:, :, None] | ((torch.any(shift_vecs != 0.0, axis=-1))[None, None, :])

    if atom_mask is not None:
        triu_mask = triu_mask & atom_mask[:, None, None] & atom_mask[None, :, None]
    if shift_mask is not None:
        triu_mask = triu_mask & shift_mask[None, None, :]

    # calculate pairwise distances
    shifted_pos = pos[:, None, :] + shift_vecs[None, :, :]
    r2 = torch.sum((pos[:, None, None, :] - shifted_pos[None, :, :, :]) ** 2, axis=-1)
    r = torch.sqrt(r2 + 1e-20)

    # calculate coordination numbers (n_atoms,)
    rco = rcov[Z][:, None] + rcov[Z][None, :]  # (n_atoms, n_atoms)
    rr = rco[:, :, None] / r  # (n_atoms, n_atoms, 1+n_shift)
    damp = torch.sigmoid(k1 * (rr - 1.0))  # (n_atoms, n_atoms, 1+n_shift)
    if cnthr is not None and cutoff_smoothing == "poly":
        damp *= poly_smoothing(r, cnthr)
    if cnthr is not None:
        damp = torch.where(r <= cnthr, damp, torch.tensor(0.0))
    damp = torch.where(triu_mask, damp, torch.tensor(0.0))
    damp = torch.sum(damp, axis=2)
    nc = torch.sum(damp, axis=1) + torch.sum(damp, axis=0)  # (n_atoms,)

    # calculate c6 and c8
    cn0 = c6ab[:, :, :, :, 0][Z, :][:, Z]
    cn1 = c6ab[Z, 1, :, 0, 1]  # (n_atoms, 5)
    cn2 = c6ab[1, Z, 0, :, 2]  # (n_atoms, 5)
    k3_rnc_1 = torch.where(cn1 >= 0.0, k3 * (nc[:, None] - cn1) ** 2, torch.tensor(-1.0e20))
    k3_rnc_2 = torch.where(cn2 >= 0.0, k3 * (nc[:, None] - cn2) ** 2, torch.tensor(-1.0e20))
    r_ratio_1 = torch.softmax(k3_rnc_1, dim=-1).to(torch.float32)
    r_ratio_2 = torch.softmax(k3_rnc_2, dim=-1).to(torch.float32)
    c6 = (cn0 * r_ratio_1[:, None, :, None] * r_ratio_2[None, :, None, :]).sum(dim=(-1,-2))
    c8 = 3 * c6 * r2r4[Z][:, None] * r2r4[Z][None, :]
    
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
        tmp = a1 * torch.sqrt(c8 / c6) + a2
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

    e68 = torch.where(r <= cutoff, e68, torch.tensor(0.0))

    e68 = torch.where(triu_mask, e68, torch.tensor(0.0))

    return torch.sum(e68.to(torch.float64).sum()) * 2.0

    #e68_same_cell = e68[:, :, 0]
    #e68_same_cell = torch.where(torch.arange(n_atoms)[:, None] < torch.arange(n_atoms)[None, :], e68_same_cell, torch.tensor(0.0))
    #e68_diff_cell = e68[:, :, 1:]
    #e_same_cell = torch.sum(e68_same_cell.to(torch.float64).sum()) * 2.0
    #e_diff_cell = torch.sum(e68_diff_cell.to(torch.float64).sum()) * 2.0
    #return e_same_cell + e_diff_cell