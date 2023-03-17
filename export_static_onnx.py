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
    mat = np.array([x.flatten(), y.flatten(), z.flatten()]).T
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

    def forward(self, Z, pos, shift_vecs, cell_volume):
        r = self.dftd_module.calc_energy(
            Z,
            pos,
            shift_vecs,
            cell_volume,
            damping=self.damping,
        )
        return r[0]["energy"]

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=str, help="number of repeats inside cell")
    parser.add_argument("--clip_num_atoms", type=int, help="max number of atoms (exceeded atoms are trashed)")
    parser.add_argument("--out_dir", type=str, help="onnx output dir", required=True)
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

    print("n_atoms = ", len(Z), "n_cell = ", len(shift_vecs), file=sys.stderr)
    print("atoms = ", atoms, file=sys.stderr)

    args = {
        "Z": Z,
        "pos": pos.type(torch.float32),
        "shift_vecs": shift_vecs.type(torch.float32),
        "cell_volume": cell_volume,
    }

    exporter = ExportONNX(cutoff=cutoff, damping="bj")

    print("energy = ", float(exporter.forward(**args)), "eV")

    print("out_dir = ", out_dir, file=sys.stderr)
    ppe_onnx.export_testcase(exporter, tuple(args.values()), out_dir, verbose=True)