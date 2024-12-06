from ase.optimize import BFGS
from ase.visualize import view
from ase import Atoms
from ase.io.trajectory import Trajectory
import numpy as np
import re
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin


class system:
    def __init__(self, stoic: str, cell: list, density: float):

        volume = np.prod(cell)  #volume from cell parameters (atoms/nm)
        tot_atoms = density*volume  #total number of atoms

        pattern = r'([A-Z][a-z]?\d*)'
        elements = []
        counts = []
        matches = re.findall(pattern, stoic)

        #seperating the stoichometry into two lists: elements and their counts
        for match in matches:
            element = ''.join([char for char in match if char.isalpha()])
            count = ''.join([char for char in match if char.isdigit()])

            if not count:
                count = 1
            else:
                count = int(count)

            elements.append(element)
            counts.append(count)

        #calculating the actual number of each element in the cell based on the density and stoichometry
        tot_count = sum(counts)
        factor = tot_atoms/tot_count
        counts = np.array(counts)
        real_nums = factor*counts
        real_nums = np.round(real_nums).astype(int)  #there's a possibility the total number of desired atoms isn't divisible, so density is approximated
        real_tot = np.sum(real_nums)

        approx_density = real_tot/volume  #in case the tot_count wasn't divisible, shows the approximated density
        sys = ''.join([f"{element}{num}" if num > 1 else f"{element}" for element, num in zip(elements, real_nums)])
        #the new system of atoms is created

        positions = np.random.rand(real_tot, 3) * np.array(cell) #random positions

        self.atoms = Atoms(sys, positions=positions, cell=cell, pbc=False)

        print("Desired Density: ", density, "atoms/nm^3")
        print("Approximated Density: ", approx_density, "atoms/nm^3")

    def peek(self):
        view(self.atoms)

    def relocate(self, fmax: float = 0.1, traj: str = "traj.traj"):
        from ase.calculators.emt import EMT
        self.atoms.calc = EMT() #what potential should I use instead?
        dyn = BFGS(self.atoms, trajectory = traj)
        dyn.run(fmax=fmax)

        trajectory = Trajectory(traj)
        return view(trajectory)


def T_high(system, temp: float, time_step: float, time: float, traj: str = "traj.traj"):
    MaxwellBoltzmannDistribution(system, temperature_K=temp)
    dyn = VelocityVerlet(system, time_step * units.fs, trajectory = traj)
    dyn.run(time/time_step)
    trajectory = Trajectory(traj)

    return system, trajectory


def T_change(system, time_step: float, time: float, temp: int, traj: str = "traj.traj"):
    dyn = Langevin(system, time_step * units.fs, temperature_K = temp, trajectory = traj, friction = 0.002)
    dyn.run(time/time_step)
    trajectory = Trajectory(traj)

    return system, trajectory


def T_low(system, time_step: float, time: float, traj: str = "traj.traj"):
    dyn = VelocityVerlet(system, time_step * units.fs, trajectory=traj)
    dyn.run(time / time_step)
    trajectory = Trajectory(traj)

    return system, trajectory



test1 = system(stoic = "CuAu3Ag2", cell = [10, 10, 10], density = 0.05)
test1.peek()  #views the system of atoms
test1.relocate(fmax = 0.2)  #runs a BFGS geometry optimization so the atoms don't have overlap

atoms, traj = T_high(test1.atoms, temp = 1700, time_step = 5, time = 10000)
view(traj)
atoms, traj = T_change(atoms, temp = 100, time_step = 5, time = 30000)
view(traj)
atoms, traj = T_low(atoms, time_step = 5, time = 10000)
view(traj)
