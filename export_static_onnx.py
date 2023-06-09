import sys

from ase.build import bulk
from ase.units import Bohr

import torch
import numpy as np

from torch_dftd.dftd3_xc_params import get_dftd3_default_params

#import pfvm.onnx
#pfvm.onnx.register_custom_operators()

import pytorch_pfn_extras.onnx as ppe_onnx
from torch_dftd_static.nn.dftd3_module import DFTD3ModuleStatic
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

def _cell_width(cell):
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

def _calc_shift_int(cell, pbc, cutoff):
    w = _cell_width(cell)
    eps = 1e-6
    n_shifts = np.ceil(2 * cutoff / w + eps)
    n_shifts = np.where(pbc, n_shifts, 1)

    x, y, z = np.meshgrid(
                np.arange(n_shifts[0]),
                np.arange(n_shifts[1]),
                np.arange(n_shifts[2]),
                indexing="ij")
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    shift_int = np.array([x, y, z]).T

    # only half of shift_int are needed
    neg_shift_int = (-shift_int) % n_shifts
    needed = (shift_int[:, 0] < neg_shift_int[:, 0]) | \
        ((shift_int[:, 0] == neg_shift_int[:, 0]) & (shift_int[:, 1] < neg_shift_int[:, 1])) | \
        ((shift_int[:, 0] == neg_shift_int[:, 0]) & (shift_int[:, 1] == neg_shift_int[:, 1]) & (shift_int[:, 2] <= neg_shift_int[:, 2]))
    shift_int = shift_int[needed]
    needs_both_ij_ji = np.any(shift_int != (neg_shift_int[needed]), axis=-1)

    # shift_int = np.where(shift_int > n_shifts // 2, shift_int - n_shifts, shift_int)

    return torch.tensor(shift_int), torch.tensor(needs_both_ij_ji), torch.tensor(n_shifts)

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
            dtype=torch.float64,
            cutoff_smoothing=cutoff_smoothing,
        )
        self.damping = damping

    def forward(self, Z, pos, shift_int, needs_both_ij_ji, n_shifts, cell, atom_mask, shift_mask):
        r = self.dftd_module.calc_energy(
            Z=Z,
            pos=pos,
            shift_int=shift_int,
            needs_both_ij_ji=needs_both_ij_ji,
            n_shifts=n_shifts,
            cell=cell,
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
    parser.add_argument("--compare_with", type=str, help="device to compare with (`pfvm` or `mncore`)", default=None)
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

    shift_int, needs_both_ij_ji, n_shifts = _calc_shift_int(cell, pbc, cutoff=cutoff)

    if args.pad_num_atoms is not None:
        atom_mask = torch.tensor(np.arange(args.pad_num_atoms) < len(Z))
        n_pad = args.pad_num_atoms - len(Z)
        assert n_pad >= 0
        Z = torch.nn.functional.pad(Z, (0, n_pad), mode="constant")
        pos = torch.nn.functional.pad(pos, (0, 0, 0, n_pad), mode="constant")
    else:
        atom_mask = torch.ones(len(Z), dtype=bool)

    if args.pad_num_cells is not None:
        shift_mask = torch.tensor(np.arange(args.pad_num_cells) < len(shift_int))
        n_pad = args.pad_num_cells - len(shift_int)
        assert n_pad >= 0
        shift_int = torch.nn.functional.pad(shift_int, (0, 0, 0, n_pad), mode="constant")
    else:
        shift_mask = torch.ones(len(shift_int), dtype=bool)

    #print(n_shifts)
    #print(shift_int)
    #print(needs_both_ij_ji)

    print("n_atoms = ", len(Z), "n_cell = ", len(shift_int), file=sys.stderr)
    print("atoms = ", atoms, file=sys.stderr)

    inputs = {
        "Z": Z,
        "pos": pos.type(torch.float64),
        "shift_int": shift_int.type(torch.float64),
        "needs_both_ij_ji": needs_both_ij_ji,
        "n_shifts": n_shifts,
        "cell": torch.tensor(cell).to(torch.float64),
        "atom_mask": atom_mask,
        "shift_mask": shift_mask,
    }

    exporter = ExportONNX(cutoff=cutoff, damping="bj")

    # compare energy with original implementation
    energy = float(exporter.forward(**inputs))
    calc_orig = TorchDFTD3Calculator(atoms=atoms, device="cpu", damping="bj", cutoff=cutoff, dtype=torch.float64)
    energy_orig = float(atoms.get_potential_energy())
    print("energy      = ", energy, "eV")
    print("energy_orig = ", energy_orig, "eV")
    assert abs(energy - energy_orig) < 1e-7 * abs(energy_orig)

    print("out_dir = ", out_dir, file=sys.stderr)
    ppe_onnx.export_testcase(exporter, tuple(inputs.values()), out_dir, verbose=True,
                             input_names=["Z","pos","shift_int","needs_both_ij_ji","n_shifts","cell","atom_mask","shift_mask"])

    if args.compare_with is not None:
        from codegen.utils import codegen_tempfile, storage
        from mncore.mndevice import get_device
        from mncore.runtime_core._context import Context, context
        from mncore.runtime_core._registry import Registry
        device = get_device(args.compare_with)
        mncore_context = Context(device, Registry())
        Context.switch_context(mncore_context)
        mncore_context.registry.register("model_dftd3", exporter)
        options = {
            "pfvm_compatible": True,
            "float_dtype": "double",
            "codegen_dir": "tmp/",
            "save_onnx": True,
        }
        print("compile options: ", options)
        with codegen_tempfile.TemporaryDirectoryWithPID() as tmpdir:
            dftd3_on_mncore, _ = mncore_context.compile(
                "dftd3",
                lambda kwargs: exporter(**kwargs),
                [],
                storage.path(tmpdir),
                inputs,
                options,
            )
        result = dftd3_on_mncore(inputs)
        energy_mncore = float(result["result"])
        print("energy (my impl.)           = ", energy, "eV")
        print("energy (original impl.)     = ", energy_orig, "eV")
        print("energy (my impl. on device) = ", energy_mncore, "eV")
        assert abs(energy_orig - energy_mncore) < 1e-6 * abs(energy_orig)
