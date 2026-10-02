# FDP: Universal machine-learning interatomic potentials on lanthanide ternaries

Code and data for the manuscript

> **Reference Conventions and Structure Types, Rather Than 4f Occupancy, Shape Lanthanide Errors in Universal Machine-Learning Interatomic Potentials**
> Jungko Moni Chakma, Md. Mynul Hasan, Mohammad Asaduzzaman Chowdhury
> Department of Mechanical Engineering, Dhaka University of Engineering and Technology (DUET), Gazipur, Bangladesh

We compare five universal machine-learning interatomic potentials (uMLIPs) with DFT formation energies for ternary compounds of a lanthanide (F), a d-block metal (D) and a p-block anion (P). The DFT reference covers 82 compounds computed with Materials Project settings. A further 945 compounds without DFT are used to measure how much the models disagree with one another.

Each model has its own notebook. All five run the same protocol: the same parent structures, optimizer, force criterion, substitution rule and formation-energy definition. Only the calculator differs.

## Models

| Model | Checkpoint | Training data | Energy convention |
|---|---|---|---|
| CHGNet | default pretrained release of `chgnet` | MPtrj | MP2020-corrected |
| MACE-MP-0 | `medium`, float64 | MPtrj | uncorrected |
| MatterSim-v1-5M | `MatterSim-v1.0.0-5M.pth` | In-house PBE/PBE+U data generated with pymatgen MPRelaxSet | uncorrected |
| GRACE-3L-OAM-L | `GRACE-3L-OMAT-large-ft-AM` | Pretrained on OMat24, fine-tuned on sAlex + MPtrj | uncorrected |
| PET-OAM-XL | `pet-oam-xl`, version 1.0.0 | Pretrained on OMat24, fine-tuned on sAlex + MPtrj | uncorrected |

The notebooks were written for Google Colab with a GPU runtime and Python 3.12. Python 3.13 and newer break part of the stack. Run each model in its own session, because their PyTorch and TensorFlow requirements conflict. Every notebook also needs:

```bash
pip install pymatgen ase phonopy seekpath pandas numpy
```

The calculator cell is the only model-specific part of each notebook. Everything after it is identical.

### CHGNet

```bash
pip install chgnet
```

```python
import torch
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
chgnet_model = CHGNet.load()
CHGNET_CALC = CHGNetCalculator(model=chgnet_model, use_device=DEVICE)
```

`CHGNet.load()` loads the default pretrained weights shipped with the installed `chgnet` release. The force cell prints the package version, so record it alongside your results.

### MACE-MP-0

```bash
pip install mace-torch
```

```python
import torch
from mace.calculators import mace_mp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MACE_CALC = mace_mp(model="medium", default_dtype="float64", device=DEVICE)
```

MACE-MP-0 runs in float64. The force cell prints the parameter dtype to confirm it took effect.

### MatterSim-v1-5M

```bash
pip install --upgrade torch torchvision torchaudio
pip install --upgrade torch_geometric
pip install mattersim
```

```python
import torch
from mattersim.forcefield import MatterSimCalculator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MATTERSIM_CALC = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=DEVICE)
```

### GRACE-3L-OAM-L

```bash
pip install tensorpotential "tensorflow[and-cuda]<=2.20" tf_keras
```

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"     # must be set before TensorFlow is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorpotential.calculator import grace_fm
GRACE_CALC = grace_fm("GRACE-3L-OMAT-large-ft-AM", mode="uniform")
```

`GRACE-3L-OMAT-large-ft-AM` is the full name of GRACE-3L-OAM-L. GRACE runs on TensorFlow, so check that TensorFlow sees the GPU (`tf.config.list_physical_devices("GPU")`) before starting.

### PET-OAM-XL

```bash
pip install upet "nvalchemi-toolkit-ops==0.3.0"
```

`nvalchemi-toolkit-ops` passes a float `max_neighbors` to `torch.full`, which fails with `full(): argument 'size' must be tuple of ints, but got float`. The second cell of the notebook fixes this by inserting `max_neighbors = int(max_neighbors)` into `nvalchemiops/.../naive.py`. Run that cell, restart the runtime, then build the calculator:

```python
from upet.calculator import UPETCalculator

DEVICE = "cuda"    # "cpu" if no GPU
PET_CALC = UPETCalculator(model="pet-oam-xl", version="1.0.0", device=DEVICE)
```

A short smoke test on bulk Cu follows to confirm the calculator works. If the XL model runs out of GPU memory, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Repository layout

```
FDP-Project/
├── notebooks/
│   ├── CHGNet.ipynb
│   ├── MACE_MP_0.ipynb
│   ├── MatterSim_v1_5M.ipynb
│   ├── GRACE_3L_OAM_L.ipynb
│   └── PET_OAM_XL.ipynb
├── data/
│   ├── cifs.zip                     # parent (seed) structures from the Materials Project
│   ├── batch_*.csv                  # candidate lists
│   ├── mp_reference_structures/     # elemental ground states not in the ASE bulk library
│   ├── Supplementary_Data_1.xlsx    # uMLIP formation energies, 945 screening compounds
│   └── Supplementary_Data_2.xlsx    # DFT vs uMLIP, 82 benchmark compounds
└── README.md
```

## Workflow

The cells run in the same order in every notebook.

### 1. Inputs
Upload `cifs.zip` (the Materials Project parent structures) and the candidate CSV.

### 2. Relax the parents with the model under test
- FIRE with `FrechetCellFilter`, full cell relaxation, `fmax = 0.005 eV/Å`.
- 500 steps for CHGNet, MACE-MP-0, MatterSim-v1-5M and PET-OAM-XL; 1000 steps for GRACE-3L-OAM-L.
- Because each model relaxes its own parents, candidates start on that model's energy surface rather than from a DFT geometry.

### 3. Index the templates
Each relaxed parent is keyed by its reduced stoichiometry signature `(F, D, P)` and its crystal system (`SpacegroupAnalyzer`, `symprec = 1e-2`).

### 4. Build the candidates
The F, D and P species of each CSV row are substituted into the parent with the matching key. Rows with no matching template are skipped and logged.

### 5. Relax the candidates
Same optimizer and criterion as step 2: `fmax = 0.005 eV/Å`, 500 steps.

### 6. Elemental references
- Each element is relaxed with the same calculator to `fmax = 0.005 eV/Å` within 2000 steps, so no model is scored against another model's elemental energies.
- Starting structures come from `ase.build.bulk`, except when a file named `<Element>.vasp` or `<Element>.cif` exists in `mp_reference_structures/`. That folder holds the Materials Project ground state (E_hull = 0) for B, N, O, P, S, Se, Mn, La and Sm.
- N and O are relaxed at fixed cell. All other elements are relaxed with full cell freedom.

### 7. Rescue pass
Candidates whose final force exceeds `2 × fmax` are restarted from their current geometry for up to 2000 more steps.

### 8. Formation energies
E_form = [E_total − Σ n_i μ_i] / N_atoms

where μ_i are the model's own elemental reference energies.

### Convergence tiers

Convergence is judged from the final maximum atomic force, not from the optimizer's `converged()` flag, because that flag reports on the filtered degrees of freedom.

| Tier | Final max force | Use |
|---|---|---|
| `converged_strict` | ≤ 0.005 eV/Å | kept |
| `near_converged` | ≤ 0.010 eV/Å | kept |
| `unconverged` | > 0.010 eV/Å | excluded from all statistics |

Each relaxation also records the space group before and after (`symprec = 1e-3`), the number of steps, and whether the step limit was reached.

## Optional modules (not used in the manuscript)

The notebooks contain two further stages. Their results are not reported in the paper.

**Force benchmark**
- At each relaxed geometry: single-point energy, forces and hydrostatic pressure.
- Three Gaussian rattles (σ = 0.03 Å per Cartesian component). Each rattle is seeded from `zlib.crc32(structure_id)`, so all five models see identical displaced structures.
- Reported: RMS, maximum and net force; the energy change on rattling; and the difference between the single-point energy and the energy the optimizer reported.

**Phonons**
- Uses converged and near-converged candidates only.
- Symmetry is refined once (`symprec = 1e-2`). The refinement is rejected if it moves any atom more than 0.10 Å.
- The structure is then re-relaxed to 1e-4 eV/Å. If it doesn't reach 5e-4 eV/Å, no phonons are reported.
- phonopy finite displacements of 0.01 Å (`is_diagonal=False`).
- Supercells have every edge ≥ 10 Å and at least 2 repetitions per axis, capped at 500 atoms and 600 displacements. The PET notebook includes a rerun cell that raises the atom cap to 1400 for six compounds that failed on memory.
- Force drift is removed and force constants are symmetrized.
- Dynamical stability is judged at the q-points commensurate with the supercell, with an imaginary tolerance of −0.05 THz.
- Thermal properties at 0, 75, 150, 300 and 600 K on a mesh of length 45. The CHGNet and MatterSim notebooks write them only for dynamically stable cells. The MACE-MP-0, GRACE-3L-OAM-L and PET-OAM-XL notebooks write them for every cell, so filter on `dynamically_stable` before using them.
- Band structures (SeeK-path paths) and phonon DOS are plotted for 12 compounds absent from the Materials Project.

## Input format

The candidate CSV has one row per structure:

| Column | Meaning |
|---|---|
| `structure_id` | Unique ID, e.g. `CeNiO4__orthorhombic` |
| `F`, `D`, `P` | Lanthanide, d-block metal, anion |
| `stoich_F`, `stoich_D`, `stoich_P` | Reduced stoichiometry signature |
| `polymorph` | Crystal system of the template (`cubic`, `hexagonal`, `monoclinic`, `orthorhombic`, `tetragonal`, `trigonal`) |

Large sets were run in batches (`batch_1.csv`, ...). The notebooks include restore and bypass cells that reload saved relaxed structures, seed logs or reference energies from earlier sessions instead of recomputing them. Skip these cells on a fresh run.

## Outputs

| Stage | CHGNet | MACE-MP-0 | MatterSim-v1-5M | GRACE-3L-OAM-L | PET-OAM-XL |
|---|---|---|---|---|---|
| Elemental references | `chgnet_element_ref_energies.json` | `mace_element_ref_energies.json` | `mattersim_element_ref_energies.json` | `grace_element_ref_energies.json` | `pet_element_ref_energies.json` |
| Formation energies | `chgnet_formation_energies.csv` | `mace_subset_formation_energies.csv` | `mattersim_formation_energies.csv` | `grace_subset_formation_energies.csv` | `pet_subset_formation_energies.csv` |
| Forces | `chgnet_subset_formation_forces.csv` | `mace_subset_formation_forces.csv` | `mattersim_subset_formation_forces.csv` | `grace_subset_formation_forces.csv` | `pet_subset_formation_forces.csv` |
| Phonons | `chgnet_phonon_55.csv` | `mace_mp_0_phonon_55.csv` | `mattersim_v1_5m_phonon_55.csv` | `grace-3l-oam-l_phonon_55.csv` | `pet-oam-xl_phonon_55.csv` |

Every notebook also writes:
- `relaxed_seeds/` with `seed_relax_log.json`
- `candidates_init/`, `candidates_relaxed/` and `candidate_meta.json`
- `phonopy_<model>/` with the force constants, and `figs_phonon_<model>/` with the figures

The formation-energy CSV has the columns `structure_id`, `energy_eV`, `e_form_per_atom`, `convergence_tier`, `final_fmax` and `flags`. The `flags` column marks values that rely on an unconverged elemental reference (`noncvg_ref:X`) or on an ASE bulk starting structure (`asebulk_ref:X`).

## Data

- **Supplementary Data 1** (`combined_MLIP_formation_energies_final.xlsx`): formation energies from all five uMLIPs for the 945 screening compounds. Of the 967 relaxed structures, 22 were removed: 11 had no CHGNet prediction and 11 had non-physical MACE-MP-0 energies (|E_form| > 5 eV/atom).
- **Supplementary Data 2** (`DFT_vs_MLIP_Comparison.xlsx`): DFT and uMLIP formation energies for the 82 benchmark compounds. It contains raw and MP2020-corrected DFT references on both the Yb_2 and Yb_3 bases, the per-compound MP2020 correction breakdown, and the correction scheme used.

### DFT reference (summary)

The 82 reference compounds were computed with VASP (PAW) using Materials Project settings:

- PBE, with PBE+U (Dudarev) only for oxides of the MP-calibrated d-block metals.
- 520 eV cutoff, spin-polarized, `LASPH = .TRUE.`, `LMAXMIX = 6`.
- Relaxation to 0.01 eV/Å, followed by a tetrahedron-method static run.
- k-point density of 1333 per reciprocal atom.

Lanthanide potentials follow the Materials Project mapping: the 4f shell is in the valence for La, Ce, Eu and Gd, and frozen in the core for the others, with Yb_2 for Yb. Full settings are in the manuscript Methods.

## Energy conventions

CHGNet predicts **MP2020-corrected** formation energies. MACE-MP-0, MatterSim-v1-5M, GRACE-3L-OAM-L and PET-OAM-XL predict **uncorrected** ones. Compare each model with the reference that matches its training convention. If you compare all five against one reference, the error of the mismatched models grows by more than a factor of three.

## Citation

If you use this code or data, please cite the manuscript (reference to be added on publication).

## Contact

Md. Mynul Hasan, hasanmynul2222.mh@gmail.com
