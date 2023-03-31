"""pytorch implementation of Grimme's D3 method"""  # NOQA
from typing import Dict, Optional

import torch
from torch import Tensor
from torch_dftd.functions.smoothing import poly_smoothing

# conversion factors used in grimme d3 code

from torch_dftd.functions.dftd3 import d3_k1, d3_k3

def _ncoord_all_pair(
    rco: Tensor,
    r: Tensor,
    cutoff: Optional[float] = None,
    k1: float = d3_k1,
    cutoff_smoothing: str = "none",
) -> Tensor:
    """Compute coordination numbers by adding an inverse damping function
    Args:
        rco: (n_atoms, n_atoms)
        r: (n_atoms, n_atoms, n_shift)
        Returns:
        g (Tensor): (n_atoms, ) coordination number for each atom
    """
    rr = rco[:, :, None] / r
    damp = torch.sigmoid(k1 * (rr - 1.0))
    if cutoff is not None and cutoff_smoothing == "poly":
        damp *= poly_smoothing(r, cutoff)
    if cutoff is not None:
        damp = torch.where(r <= cutoff, damp, torch.tensor(0.0))
    damp = torch.where(r >= 1e-8, damp, torch.tensor(0.0))  # remove self-edge

    return torch.sum(damp, axis=[1, 2])

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
):
    # calculate coordination numbers (n_atoms,)
    shift_vecs_half = shift_vecs
    shift_vecs_all = torch.cat((shift_vecs_half, -shift_vecs_half, torch.zeros((1, 3), dtype=shift_vecs_half.dtype)), 0)

    diff = pos[None, :, :] - pos[:, None, :]  # (n_atoms, n_atoms, 3)
    r2 = torch.sum((diff[:, :, None, :] + shift_vecs_all[None, None, :, :]) ** 2, axis=-1)  # (n_atoms, n_atoms, n_shift)
    r = torch.sqrt(r2 + 1e-20)  # (n_atoms, n_atoms, n_shift)

    rco = rcov[Z][:, None] + rcov[Z][None, :]  # (n_atoms, n_atoms)

    nc = _ncoord_all_pair(
        rco,
        r,
        cutoff=cnthr,
        cutoff_smoothing=cutoff_smoothing,
        k1=k1,
    )

    # calculate energy
    def energy(v, idx_i, idx_j):
        n_pair = idx_i.shape[0]
        Z_pair = Z[idx_i] * 95 + Z[idx_j]

        cn0 = c6ab[:, :, :, :, 0].view(95*95, 5*5)[Z_pair]
        cn1 = c6ab[:, :, :, 0, 1].view(95*95, 5)[Z_pair]
        cn2 = c6ab[:, :, 0, :, 2].view(95*95, 5)[Z_pair]
        r_cn = ((cn1[:, :, None] - nc[idx_i, None, None]) ** 2 + (cn2[:, None, :] - nc[idx_j, None, None]) ** 2).view(n_pair, 5*5)
        k3_rnc = torch.where(cn0 > 0.0, k3 * r_cn, torch.tensor(-1.0e20))
        r_ratio = torch.softmax(k3_rnc, dim=-1)
        c6 = (r_ratio * cn0).sum(dim=-1)
        c8 = 3 * c6 * r2r4[Z][idx_i] * r2r4[Z][idx_j]
        s6 = params["s6"]
        s8 = params["s18"]

        diff = pos[idx_i] - pos[idx_j]
        r2 = torch.sum((diff[:, None] + v[None, :]) ** 2, axis=-1)
        r = torch.sqrt(r2 + 1e-20)
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
            e6 = 1 / (r6 + tmp6[:, None])
            e8 = 1 / (r8 + tmp8[:, None])
        elif damping == "zero":
            rs6 = params["rs6"]
            rs8 = params["rs18"]
            alp = params["alp"]
            alp6 = alp
            alp8 = alp + 2.0
            tmp2 = r0ab
            rr = tmp2[:, None] / r
            damp6 = 1.0 / (1.0 + 6.0 * (rs6 * rr) ** alp6)
            damp8 = 1.0 / (1.0 + 6.0 * (rs8 * rr) ** alp8)
            e6 = damp6 / r6
            e8 = damp8 / r8
        elif damping == "zerom":
            rs6 = params["rs6"]
            rs8 = params["rs18"]
            alp = params["alp"]
            alp6 = alp
            alp8 = alp + 2.0
            tmp2 = r0ab
            r0_beta = rs8 * tmp2
            rr = r / tmp2[:, None]
            tmp = rr / rs6 + r0_beta
            damp6 = 1.0 / (1.0 + 6.0 * tmp ** (-alp6))
            tmp = rr + r0_beta
            damp8 = 1.0 / (1.0 + 6.0 * tmp ** (-alp8))
            e6 = damp6 / r6
            e8 = damp8 / r8
        else:
            raise ValueError(f"[ERROR] Unexpected value damping={damping}")

        e6 = -0.5 * s6 * c6[:, None] * e6
        e8 = -0.5 * s8 * c8[:, None] * e8
        e68 = e6 + e8

        if cutoff is not None and cutoff_smoothing == "poly":
            e68 *= poly_smoothing(r, cutoff)

        e68 = torch.where(r <= cutoff, e68, torch.tensor(0.0))

        return torch.sum(e68.to(torch.float64).sum()) * 2.0

    n_atoms = len(Z)
    triu_i, triu_j = torch.triu_indices(n_atoms, n_atoms, 1)

    E_triu = energy(shift_vecs_all, triu_i, triu_j)
    E_diag = energy(shift_vecs_half, torch.arange(n_atoms), torch.arange(n_atoms))
    # print(d3_autoev * E_triu, d3_autoev * E_diag)
    return E_triu + E_diag