from mace.calculators import mace_mp
from ase import build
import numpy as np
from ase.build import bulk
from ase.visualize import view
from ase.optimize import BFGS

calc = mace_mp(model="/Users/eraybaykal/Downloads/MACE_MPtrj_2022.9.model", dispersion=False, default_dtype="float64", device='cpu')

lattice = bulk('Cu', 'fcc', a=1.0, cubic=True)
view(lattice)

lattice.calc = calc
print(lattice.get_potential_energy())

dyn = BFGS(lattice)
dyn.run(fmax=0.05)

print(lattice.get_potential_energy())
view(lattice)

