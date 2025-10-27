from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
#******* Fixed voltage routines
from Fixed_Voltage_routines import *
#******* exclusions routines
from electrode_sapt_exclusions import *
#***********
import random
import numpy
import subprocess


class MM(object):
    def __init__(self, pdb_list , residue_xml_list , ff_xml_list , **kwargs  ):
          self.temperature = 300*kelvin
          self.temperature_drude = 1*kelvin
          self.friction = 1/picosecond
          self.friction_drude = 1/picosecond
          self.timestep = 0.001*picoseconds
          self.small_threshold = 1e-6
          self.cutoff = 1.4*nanometer  
          self.QMMM = False
          self._electrolyte_indices_array = numpy.array([], dtype=numpy.int64)

          if 'temperature' in kwargs : self.temperature = kwargs['temperature']
          if 'cutoff' in kwargs : self.cutoff = kwargs['cutoff']
          if 'QMregion_list' in kwargs :
              self.QMMM = True
              self.QMregion_list = kwargs['QMregion_list'] 
           
          for residue_file in residue_xml_list:
               Topology().loadBondDefinitions(residue_file)

          self.pdb = PDBFile( pdb_list[0] )
          self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
          self.forcefield = ForceField(*ff_xml_list)
          self.modeller.addExtraParticles(self.forcefield)

          if self.QMMM : self.modeller.topology.addQMatoms( self.QMregion_list )

          self.polarization = (self.pdb.topology.getNumAtoms() != self.modeller.topology.getNumAtoms())

          if self.polarization :
              self.integrator = DrudeLangevinIntegrator(self.temperature, self.friction, self.temperature_drude, self.friction_drude, self.timestep)
              self.integrator.setMaxDrudeDistance(0.02)
          else :
              self.integrator = LangevinIntegrator(self.temperature, self.friction, self.timestep)

          self.system = self.forcefield.createSystem(self.modeller.topology, nonbondedCutoff=self.cutoff, constraints=HBonds, rigidWater=True)
          self.nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
          self.customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]

          self.nbondedForce.setNonbondedMethod(NonbondedForce.PME)
          self.customNonbondedForce.setNonbondedMethod(min(self.nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))

    def set_trajectory_output( self, filename , write_frequency ):
          self.simmd.reporters = []
          self.simmd.reporters.append(DCDReporter(filename, write_frequency))

    def set_periodic_residue(self, flag):
          for i, f in enumerate(self.system.getForces()):
               f.setForceGroup(i)
               if flag and type(f) in [HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, RBTorsionForce]:
                    f.setUsesPeriodicBoundaryConditions(True)

    def setPMEParameters( self , pme_alpha , pme_grid_a , pme_grid_b , pme_grid_c ):
        self.nbondedForce.setPMEParameters( pme_alpha , pme_grid_a , pme_grid_b , pme_grid_c )

    def set_platform( self, platformname ):
          try:
              self.platform = Platform.getPlatformByName(platformname)
          except Exception:
              print(f"Error: Could not find platform '{platformname}'. Available platforms: {[Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]}")
              sys.exit(1)

          self.properties = {}
          if self.QMMM and platformname != 'Reference':
              print('Can only run QM/MM simulation with reference platform !')
              sys.exit()
          elif platformname == 'Reference' and self.QMMM:
              self.properties = {'ReferenceVextGrid': 'true'}
          elif platformname == 'CUDA':
              self.properties = {'Precision': 'mixed'}

          self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
          self.simmd.context.setPositions(self.modeller.positions)

    def initialize_electrodes( self, Voltage, cathode_identifier , anode_identifier , chain=False, exclude_element=(), **kwargs ):
        self.Cathode = Electrode_Virtual( cathode_identifier , "cathode" , Voltage , self , chain , exclude_element )
        self.Anode   = Electrode_Virtual( anode_identifier   , "anode"   , Voltage , self , chain , exclude_element )
        self.Conductor_list = []

        state = self.simmd.context.getState(getPositions=True)
        positions = state.getPositions()
        boxVecs = self.simmd.topology.getPeriodicBoxVectors()
        self.set_electrochemical_cell_parameters( positions, boxVecs )

        self.Cathode.initialize_Charge( self.Lgap, self.Lcell, self )
        self.Anode.initialize_Charge( self.Lgap, self.Lcell, self )

        self._cathode_indices = numpy.array([atom.atom_index for atom in self.Cathode.electrode_atoms], dtype=numpy.int64)
        self._anode_indices = numpy.array([atom.atom_index for atom in self.Anode.electrode_atoms], dtype=numpy.int64)

    def set_electrochemical_cell_parameters( self , positions , boxVecs ):
        z_cath = positions[self.Cathode.electrode_atoms[0].atom_index][2].value_in_unit(nanometer)
        z_anod = positions[self.Anode.electrode_atoms[0].atom_index][2].value_in_unit(nanometer)
        self.Cathode.set_z_pos(z_cath)
        self.Anode.set_z_pos(z_anod)
        self.Lcell = abs(z_cath - z_anod)
        self.Lgap = boxVecs[2][2].value_in_unit(nanometer) - self.Lcell

    def initialize_electrolyte( self , Natom_cutoff=100):
        electrolyte_names=set()
        self.electrolyte_atom_indices=[]
        for res in self.simmd.topology.residues():
            if res.name in electrolyte_names or len(res._atoms) < Natom_cutoff:
                if res.name not in electrolyte_names: electrolyte_names.add(res.name)
                self.electrolyte_atom_indices.extend([atom.index for atom in res._atoms])
        self._cache_electrolyte_charges()

    def _cache_electrolyte_charges(self):
        self._electrolyte_charges = numpy.array([self.nbondedForce.getParticleParameters(idx)[0]._value for idx in self.electrolyte_atom_indices], dtype=numpy.float64)
        self._electrolyte_indices_array = numpy.array(self.electrolyte_atom_indices, dtype=numpy.int64)

    def _get_sim_state(self, get_forces=True, get_positions=True):
        state = self.simmd.context.getState(getForces=get_forces, getPositions=get_positions)
        forces_z = state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/nanometer)[:, 2] if get_forces else None
        positions_z = state.getPositions(asNumpy=True).value_in_unit(nanometer)[:, 2] if get_positions else None
        return forces_z, positions_z

    def _calculate_new_charges(self, forces_z, indices, q_old, prefactor, voltage_term, sign):
        ez_external = numpy.zeros_like(q_old)
        mask = numpy.abs(q_old) > (0.9 * self.small_threshold)
        valid_indices = indices[mask]
        if valid_indices.size > 0:
            ez_external[mask] = forces_z[valid_indices] / q_old[mask]
        q_new = prefactor * (voltage_term + ez_external)
        q_new[numpy.abs(q_new) < self.small_threshold] = self.small_threshold * sign
        return q_new

    def Poisson_solver_fixed_voltage(self, Niterations=3, enable_warmstart=False, verify_interval=100):
        if not hasattr(self, '_warmstart_call_counter'): self._warmstart_call_counter = 0
        self._warmstart_call_counter += 1
        use_warmstart_this_step = enable_warmstart and hasattr(self, '_warm_start_cathode_q')
        if verify_interval > 0 and self._warmstart_call_counter % verify_interval == 0:
            use_warmstart_this_step = False

        _, positions_z = self._get_sim_state(get_forces=False)
        self.Cathode.compute_Electrode_charge_analytic(self, positions_z, self.Conductor_list, z_opposite=self.Anode.z_pos)
        self.Anode.compute_Electrode_charge_analytic(self, positions_z, self.Conductor_list, z_opposite=self.Anode.z_pos)

        cathode_prefactor = (2.0 / (4.0 * numpy.pi)) * self.Cathode.area_atom * conversion_KjmolNm_Au
        anode_prefactor = (-2.0 / (4.0 * numpy.pi)) * self.Anode.area_atom * conversion_KjmolNm_Au

        cathode_q = self._warm_start_cathode_q if use_warmstart_this_step else numpy.array([atom.charge for atom in self.Cathode.electrode_atoms])
        anode_q = self._warm_start_anode_q if use_warmstart_this_step else numpy.array([atom.charge for atom in self.Anode.electrode_atoms])

        for _ in range(Niterations):
            forces_z, _ = self._get_sim_state(get_positions=False)
            cathode_q = self._calculate_new_charges(forces_z, self._cathode_indices, cathode_q, cathode_prefactor, self.Cathode.Voltage / self.Lgap, 1.0)
            anode_q = self._calculate_new_charges(forces_z, self._anode_indices, anode_q, anode_prefactor, self.Anode.Voltage / self.Lgap, -1.0)
            for i, atom in enumerate(self.Cathode.electrode_atoms): atom.charge = cathode_q[i]
            for i, atom in enumerate(self.Anode.electrode_atoms): atom.charge = anode_q[i]
            self.Scale_charges_analytic_general()

        self._update_charges_in_context()
        if enable_warmstart:
            self._warm_start_cathode_q = cathode_q.copy()
            self._warm_start_anode_q = anode_q.copy()

    def _update_charges_in_context(self):
        for atom in self.Cathode.electrode_atoms: self.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, 1.0, 0.0)
        for atom in self.Anode.electrode_atoms: self.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, 1.0, 0.0)
        self.nbondedForce.updateParametersInContext(self.simmd.context)

    def Scale_charges_analytic_general(self , print_flag = False ):
        self.Cathode.Scale_charges_analytic(self, print_flag)
        self.Anode.Scale_charges_analytic(self, print_flag)

    def write_electrode_charges( self, chargeFile ):
        charges = [atom.charge for atom in self.Cathode.electrode_atoms]
        charges.extend(atom.charge for atom in self.Anode.electrode_atoms)
        chargeFile.write(" ".join(f"{q:f}" for q in charges) + "\n")
        chargeFile.flush()

    # ... Other methods like generate_exclusions, MC_Barostat_step etc. would be here ...
    def generate_exclusions(self, flag_SAPT_FF_exclusions=True):
        pass # Keeping it simple for now

class MC_parameters(object):
    def __init__( self , temperature , celldim , electrode_move="Anode" , pressure = 1.0*bar , barofreq = 25 , shiftscale = 0.2 ):
        self.RT = BOLTZMANN_CONSTANT_kB * temperature * AVOGADRO_CONSTANT_NA     
        self.pressure = pressure*celldim[0] * celldim[1] * AVOGADRO_CONSTANT_NA
        self.electrode_move, self.barofreq, self.shiftscale = electrode_move, barofreq, shiftscale
        self.ntrials, self.naccept = 0, 0
