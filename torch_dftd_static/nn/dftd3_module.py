import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from ase.units import Bohr
from torch import Tensor
from torch_dftd.functions.dftd3 import d3_autoev
from torch_dftd_static.functions.dftd3 import edisp as edisp_notriu
from torch_dftd_static.functions.dftd3_triu import edisp as edisp_triu

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
        damping: str = "zero",
        triu: bool = False,
    ) -> Tensor:
        """Forward computation to calculate atomic wise dispersion energy"""
        
        # TODO: add interface for force and stress
        
        edisp = edisp_triu if triu else edisp_notriu
        E_disp = d3_autoev * edisp(
            Z,
            pos = pos / Bohr,
            shift_vecs = shift_vecs / Bohr,
            c6ab=self.c6ab,
            r0ab=self.r0ab,
            rcov=self.rcov,
            r2r4=self.r2r4,
            params=self.params,
            cutoff=self.cutoff / Bohr,
            cnthr=self.cnthr / Bohr,
            cutoff_smoothing=self.cutoff_smoothing,
            damping=damping,
        )
        return [{"energy": E_disp}]
