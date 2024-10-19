import numpy as np
import scipy
import matplotlib.pyplot as plt

from ase.build import bulk
from ase.visualize import view

from psiflow.geometry import Geometry
from psiflow.sampling import optimize
from psiflow.hamiltonians import MACEHamiltonian


lattice = bulk('Cu', 'fcc', a=3.6, cubic=True)
view(lattice)

geometry = Geometry.from_atoms(lattice)

mace = MACEHamiltonian.mace_mp0()

minimum_geometry = optimize(
    geometry,
    mace,
    ftol=1e-4,
)                   

energy = mace.compute(geometry, 'energy')
minimum_energy = mace.compute(minimum_geometry, 'energy')
