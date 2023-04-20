import sys

from ase.build import bulk
from ase.units import Bohr

import torch
import numpy as np

from torch_dftd.dftd3_xc_params import get_dftd3_default_params

import pfvm.onnx
pfvm.onnx.register_custom_operators()

import pytorch_pfn_extras.onnx as ppe_onnx
from torch_dftd_static.nn.dftd3_module import DFTD3ModuleStatic
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

def cell_width(cell):
    xn = np.cross(cell[1, :], cell[2, :])
    yn = np.cross(cell[2, :], cell[0, :])
    zn = np.cross(cell[0, :], cell[1, :])
    xn = xn / np.linalg.norm(xn)
    yn = yn / np.linalg.norm(yn)
    zn = zn / np.linalg.norm(zn)

    cell_dx = np.abs(np.matmul(cell[0, :], xn))
    cell_dy = np.abs(np.matmul(cell[1, :], yn))
    cell_dz = np.abs(np.matmul(cell[2, :], zn))

    return np.array([cell_dx, cell_dy, cell_dz])

def calc_shift_vecs(cell, pbc, cutoff):
    w = cell_width(cell)
    n_shifts = np.ceil(cutoff / w)
    n_shifts = np.where(pbc, n_shifts, 0)

    x, y, z = np.meshgrid(
                np.arange(-n_shifts[0], n_shifts[0] + 1),
                np.arange(-n_shifts[1], n_shifts[1] + 1),
                np.arange(-n_shifts[2], n_shifts[2] + 1),
                indexing="ij")
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    
    # take triu part only
    cond = (x > 0) | ((x == 0) & (y > 0)) | ((x == 0) & (y == 0) & (z > 0))
    x, y, z = x[cond], y[cond], z[cond]

    mat = np.array([x, y, z]).T
    mat = np.concatenate(([[0, 0, 0]], mat), axis=0)
    return mat.dot(cell)

class ExportONNX(torch.nn.Module):
    def __init__(
            self,
            cutoff,
            damping: str = "zero",
            xc: str = "pbe",
            old: bool = False,
            cnthr: float = 40.0 * Bohr,
            cutoff_smoothing: str = "none",
        ):
        super().__init__()

        self.params = get_dftd3_default_params(damping, xc, old=old)
        
        self.dftd_module = DFTD3ModuleStatic(
            self.params,
            cutoff=cutoff,
            cnthr=cnthr,
            dtype=torch.float32,
            cutoff_smoothing=cutoff_smoothing,
        )
        self.damping = damping

    def forward(self, Z, pos, shift_vecs, cell_volume, atom_mask, shift_mask):
        r = self.dftd_module.calc_energy(
            Z,
            pos,
            shift_vecs,
            cell_volume,
            damping=self.damping,
            atom_mask=atom_mask,
            shift_mask=shift_mask,
        )
        return r[0]["energy"]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=str, help="number of repeats inside cell")
    parser.add_argument("--clip_num_atoms", type=int, help="max number of atoms (exceeded atoms are trashed)")
    parser.add_argument("--out_dir", type=str, help="onnx output dir", required=True)
    parser.add_argument("--pad_num_atoms", type=int, help="num_atoms after padding", required=False)
    parser.add_argument("--pad_num_cells", type=int, help="num_cells after padding", required=False)
    return parser.parse_args()

def prepare_data(args):
    atoms = bulk("Pt")
    if args.repeat is not None:
        repeat = [int(r) for r in args.repeat.split(",")]
        if len(repeat) == 1:
            repeat = repeat[0]
        atoms = atoms.repeat(repeat)
    if args.clip_num_atoms is not None:
        atoms = atoms[:args.clip_num_atoms]
    return atoms

if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir
    cutoff = 40.0 * Bohr

    atoms = prepare_data(args)

    pos = torch.tensor(atoms.get_positions())
    Z = torch.tensor(atoms.get_atomic_numbers())
    pbc = torch.tensor(atoms.pbc, dtype=bool)
    if np.any(atoms.pbc):
        cell = np.array(atoms.get_cell())
    else:
        cell = np.eye(3)
    cell_volume = np.abs(np.linalg.det(cell))

    shift_vecs = calc_shift_vecs(cell, pbc, cutoff=cutoff)
    shift_vecs = torch.tensor(shift_vecs)

    if args.pad_num_atoms is not None:
        atom_mask = torch.tensor(np.arange(args.pad_num_atoms) < len(Z))
        n_pad = args.pad_num_atoms - len(Z)
        assert n_pad >= 0
        Z = torch.nn.functional.pad(Z, (0, n_pad), mode="constant")
        pos = torch.nn.functional.pad(pos, (0, 0, 0, n_pad), mode="constant")
    else:
        atom_mask = torch.ones(len(Z), dtype=bool)

    if args.pad_num_cells is not None:
        shift_mask = torch.tensor(np.arange(args.pad_num_cells) < len(shift_vecs))
        n_pad = args.pad_num_cells - len(shift_vecs)
        assert n_pad >= 0
        shift_vecs = torch.nn.functional.pad(shift_vecs, (0, 0, 0, n_pad), mode="constant")
    else:
        shift_mask = torch.ones(len(shift_vecs), dtype=bool)

    print("n_atoms = ", len(Z), "n_cell = ", len(shift_vecs), file=sys.stderr)
    print("atoms = ", atoms, file=sys.stderr)

    args = {
        "Z": Z,
        "pos": pos.type(torch.float32),
        "shift_vecs": shift_vecs.type(torch.float32),
        "cell_volume": cell_volume,
        "atom_mask": atom_mask,
        "shift_mask": shift_mask,
    }

    exporter = ExportONNX(cutoff=cutoff, damping="bj")

    # compare energy with original implementation
    energy = float(exporter.forward(**args))
    calc_orig = TorchDFTD3Calculator(atoms=atoms, device="cpu", damping="bj", cutoff=cutoff)
    energy_orig = float(atoms.get_potential_energy())
    print("energy      = ", energy, "eV")
    print("energy_orig = ", energy_orig, "eV")
    assert abs(energy - energy_orig) < 1e-7 * abs(energy_orig)

    print("out_dir = ", out_dir, file=sys.stderr)
    ppe_onnx.export_testcase(exporter, tuple(args.values()), out_dir, verbose=True,
                             input_names=["Z","pos","shift_vecs","cell_volume","atom_mask","shift_mask"])