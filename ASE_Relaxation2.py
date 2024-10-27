from mace.calculators import mace_mp
from ase.spacegroup import crystal
from ase import Atoms
from ase.optimize import BFGS
from ase.visualize import view
from ase.io.trajectory import Trajectory
from ase.calculators.espresso import Espresso, EspressoProfile
import covalent as ct

pseudopotentials = {"Cu": "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf", "Cl": "cl_pbe_v1.4.uspp.F.UPF"}


class system:
    def __init__(self, formula: str, basis: list, spacegroup: int,
                 cell: list, size: list, calculator: str):

        self.formula = formula
        self.basis = basis
        self.spacegroup = spacegroup
        self.cell = cell
        self.size = size
        self.calculator = calculator
        self.lattice = Atoms()
        self.trajectories = []
        self.pseudopotentials = {"Cu": "Cu.paw.z_11.ld1.psl.v1.0.0-low.upf"}

        # want to make dependent variables so I don't have to keep running an update() fuction every time the variables change

    # updates the lattice system when a variable changes
    def update(self):
        from ase.spacegroup import crystal

        if self.calculator == "MACE":
            calc = mace_mp(model="/Users/eraybaykal/PycharmProjects/MD/MACE_Models/MACE_MPtrj_2022.9.model",
                           dispersion=False, default_dtype="float64")
        elif self.calculator == "ESPRESSO":
            '''
            print("hi")

            profile = EspressoProfile(
                command='/Users/eraybaykal/Downloads/qe-7.3.1/bin', pseudo_dir='//Users/eraybaykal/PycharmProjects/MD/SSSP_1.3.0_PBE_efficiency'
            )
            calc = Espresso(profile=profile, pseudopotentials=pseudopotentials)
            '''
        else:
            raise ValueError("Invalid Calculator: must be MACE or ESPRESSO")

        crystal = crystal(self.formula, self.basis, spacegroup=self.spacegroup, cellpar=self.cell,
                          size=self.size)
        self.lattice = Atoms(crystal.get_chemical_symbols(), positions=crystal.get_positions())
        self.lattice.calc = calc

    # to see the lattice, want to make it so it shows all the other initalizations as well
    def peek_init(self):
        self.update()
        view(self.lattice)

    # to watch the trajectory plot, the trajectory list will contain initalizations specific to each sim as well
    # this way I can run many different simulations parallel and compare results at the end??
    def watch(self, name: str = "blank"):
        if name == "blank":
            traj = self.trajectories[-1]
        else:
            traj = Trajectory(name)

        try:
            view(traj)
        except ValueError:
            print("no trajectories to see")

    def add_run(self, traj: Trajectory):
        self.trajectories.append(traj)


def run(cur_system: system, name: str, fmax: float = 0.04):
    cur_system.update()
    dyn = BFGS(cur_system.lattice, trajectory=name)
    dyn.run(fmax=fmax)

    cur_system.add_run(Trajectory(name))



new_system = system('CuAl', [(0.25, 0.25, 0.25), (0.15, 0.15, 0.15)], 203, [9.04, 9.04, 9.04, 90, 90, 90], (1, 1, 1),
                    "MACE")

new_system.peek_init()
run(new_system, "test.traj", 0.1)


new_system.watch()




#view(ct.get_result(dispatch_id))
#new_system.watch()


