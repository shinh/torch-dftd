import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from ase.units import Bohr
from torch import Tensor
from torch_dftd.functions.dftd3 import d3_autoang, d3_autoev
from torch_dftd_static.functions.dftd3 import edisp as edisp_triu


"""
Check that c6ab array (shape = (95,95,5,5,3)) has the following structure,
as assumed in edisp function:
- c6ab[..., 1] is constant along axis 1 (Z of second atom) and 3,
  except for Z=0 which does not represent valid atom.
  Second condition in torch.all(...) below exists because there can be
  rows/columns without valid values (because not every pair of atoms has full 5x5 table),
  and such missing values are represented by -1.
- c6ab[..., 2] is constant along axis 0 (Z of first atom) and 2, except for Z=0.

https://docs.google.com/presentation/d/15J3jDALiD_tDPT9DVi2GcIBTMUT8QlfQe9ET4iuCE0M/edit#slide=id.g226c3535966_0_34
"""
def _check_c6ab_structure(c6ab):
    assert torch.all((c6ab[:, 1:, :, :, 1] == c6ab[:, 1:2, :, 0:1, 1]) | (c6ab[:, 1:, :, :, 0] < 0)), "c6ab[..., 1] is not constant along row"
    assert torch.all((c6ab[1:, :, :, :, 2] == c6ab[1:2, :, 0:1, :, 2]) | (c6ab[1:, :, :, :, 0] < 0)), "c6ab[..., 2] is not constant along column"

class DFTD3ModuleStatic(torch.nn.Module):
    """DFTD3ModuleStatic
    Args:
        params (dict): xc-dependent parameters. alp, s6, rs6, s18, rs18.
        cutoff (float): cutoff distance in angstrom. Default value is 95bohr := 50 angstrom.
        cnthr (float): coordination number cutoff distance in angstrom.
            Default value is 40bohr := 21 angstrom.
        abc (bool): ATM 3-body interaction
        dtype (dtype): internal calculation is done in this precision.
        bidirectional (bool): calculated `edge_index` is bidirectional or not.
    """

    def __init__(
        self,
        params: Dict[str, float],
        cutoff: float = 95.0 * Bohr,
        cnthr: float = 40.0 * Bohr,
        abc: bool = False,
        dtype=torch.float32,
        cutoff_smoothing: str = "none",
    ):
        super(DFTD3ModuleStatic, self).__init__()

        if abc:
            raise ValueError("currently abc (3-body term) is not available in static version. ")

        # relative filepath to package folder
        d3_filepath = str(Path(os.path.abspath(__file__)).parent / "params" / "dftd3_params.npz")
        d3_params = np.load(d3_filepath)
        c6ab = torch.tensor(d3_params["c6ab"], dtype=dtype)
        r0ab = torch.tensor(d3_params["r0ab"], dtype=dtype)
        rcov = torch.tensor(d3_params["rcov"], dtype=dtype)
        r2r4 = torch.tensor(d3_params["r2r4"], dtype=dtype)
        # (95, 95, 5, 5, 3) c0, c1, c2 for coordination number dependent c6ab term.
        self.register_buffer("c6ab", c6ab)
        self.register_buffer("r0ab", r0ab)  # atom pair distance (95, 95)
        self.register_buffer("rcov", rcov)  # atom covalent distance (95)
        self.register_buffer("r2r4", r2r4)  # (95,)

        _check_c6ab_structure(c6ab)

        if cnthr > cutoff:
            print(
                f"WARNING: cnthr {cnthr} is larger than cutoff {cutoff}. "
                f"cutoff distance is used for cnthr"
            )
            cnthr = cutoff
        self.params = params
        self.cutoff = cutoff
        self.cnthr = cnthr
        self.dtype = dtype
        self.cutoff_smoothing = cutoff_smoothing

    def calc_energy(
        self,
        Z: Tensor,
        pos: Tensor,
        shift_vecs: Tensor,
        cell_volume: float,
        damping: str,
        atom_mask: Optional[Tensor] = None,
        shift_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward computation to calculate atomic wise dispersion energy"""
        
        # TODO: add interface for force and stress
        
        E_disp = d3_autoev * edisp_triu(
            Z,
            pos = pos / d3_autoang,
            shift_vecs = shift_vecs / d3_autoang,
            c6ab=self.c6ab,
            r0ab=self.r0ab,
            rcov=self.rcov,
            r2r4=self.r2r4,
            params=self.params,
            cutoff=self.cutoff / Bohr,
            cnthr=self.cnthr / Bohr,
            cutoff_smoothing=self.cutoff_smoothing,
            damping=damping,
            atom_mask=atom_mask,
            shift_mask=shift_mask,
        )
        return [{"energy": E_disp}]
