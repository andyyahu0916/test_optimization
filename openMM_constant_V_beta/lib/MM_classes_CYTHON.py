from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
#******* Fixed voltage routines
# 🚀 CYTHON: Import from CYTHON routines (NOT OPTIMIZED!)
try:
    from .Fixed_Voltage_routines_CYTHON import *
except ImportError:
    from Fixed_Voltage_routines_CYTHON import *
#******* exclusions routines
try:
    from .electrode_sapt_exclusions import *
except ImportError:
    from electrode_sapt_exclusions import *
#***********
import random
import numpy
import subprocess
# 🚀 OPTIMIZATION: Numba for MC Barostat ONLY (1000+ residues)
from numba import njit, prange

# 🔥 Try to import Cython module (fallback to NumPy if not available)
try:
    import electrode_charges_cython as ec_cython
    CYTHON_AVAILABLE = True
    print("✅ Cython module loaded successfully!")
except ImportError:
    CYTHON_AVAILABLE = False
    print("⚠️  Cython module not found. Run: python setup_cython.py build_ext --inplace")
    print("    Falling back to NumPy implementation.")

# Conversion factors (defined once!)
conversion_nmBohr = 18.8973
conversion_KjmolNm_Au = conversion_nmBohr / 2625.5


#*************************************************
# 🚀 CRITICAL OPTIMIZATION: Numba for MC Barostat (ONLY for 1000+ residues)
# 這是唯一真正需要 Numba 並行化的地方
#
# IMPORTANT: This function replicates the EXACT logic of the original code:
# - Uses FIRST atom as reference (not COM)
# - Scales reference position in Z direction only
# - Preserves intra-molecular geometry
#*************************************************
@njit(parallel=True, fastmath=True, cache=True)
def update_electrolyte_positions_numba(newpos, oldpos, residue_first_atoms, residue_atom_counts,
                                       reference_electrode_z, Lcell_ratio):
    """
    並行更新電解質分子位置 (用於 MC Barostat)
    
    複製原始算法:
    1. 使用每個 residue 的第一個原子作為參考點
    2. 計算分子內相對向量 (保持分子幾何)
    3. 僅在 Z 方向縮放參考點位置
    4. 重建所有原子位置
    
    適用: 1000+ residues 的大規模並行計算
    
    Parameters:
    - newpos: 新位置陣列 (將被修改)
    - oldpos: 舊位置陣列 (輸入)
    - residue_first_atoms: 每個 residue 第一個原子索引
    - residue_atom_counts: 每個 residue 原子數
    - reference_electrode_z: 固定電極的 z 座標
    - Lcell_ratio: Lcell_new / Lcell_old
    """
    n_residues = len(residue_first_atoms)
    
    for i in prange(n_residues):
        first_atom_idx = residue_first_atoms[i]
        n_atoms = residue_atom_counts[i]
        
        # Step 1: Get first atom position as reference (🔥 修正: 必須來自 oldpos)
        ref_x = oldpos[first_atom_idx, 0]
        ref_y = oldpos[first_atom_idx, 1]
        ref_z = oldpos[first_atom_idx, 2]
        
        # Step 2: Scale reference Z-coordinate (matches original line 710-714)
        # Convert to electrode-relative coordinates
        ref_z_relative = ref_z - reference_electrode_z
        # Scale by cell ratio
        ref_z_relative_scaled = ref_z_relative * Lcell_ratio
        # Convert back to global coordinates
        ref_z_new = ref_z_relative_scaled + reference_electrode_z
        
        # Step 3: Update all atoms maintaining intra-molecular vectors
        for j in range(n_atoms):
            atom_idx = first_atom_idx + j
            
            # 🔥 修正: 計算 intra-molecular vector 必須使用 oldpos
            # 從 OLD POS 計算舊的幾何結構的 Delta
            dx = oldpos[atom_idx, 0] - ref_x
            dy = oldpos[atom_idx, 1] - ref_y
            dz = oldpos[atom_idx, 2] - ref_z
            
            # 🔥 修正: 將 "舊的" delta 應用到 "新的" 參考點位置上
            # (新的參考點 X, Y 不變, 只有 Z 改變了)
            newpos[atom_idx, 0] = ref_x + dx
            newpos[atom_idx, 1] = ref_y + dy
            newpos[atom_idx, 2] = ref_z_new + dz


#*************************************************
# This MM class is meant to be very general/versatile for a range of simulation types.
# Currently, the three custom types of simulations it allows are:
#  1) QM/MM turned on by inputing "QMregion_list"  -- must use compiled version of customized OpenMM code
#  2) Fixed-Voltage MD for Supercapacitors with a variety of electrode types
#  3) Monte Carlo equilibration of liquid/solid interfaces (e.g. electrode/electrolyte)
#
#**************************************************
class MM(object):
    # input to init is 3 lists , list of pdb files, list of residue xml files, list of force field xml files .
    # **kwargs input is used to override default settings ...
    def __init__(self, pdb_list , residue_xml_list , ff_xml_list , **kwargs  ):
          #*************************************
          #  DEFAULT RUN PARAMETERS, to overide defaults input  as **kwargs ...
          #**************************************
          self.temperature = 300*kelvin
          self.temperature_drude = 1*kelvin
          self.friction = 1/picosecond
          self.friction_drude = 1/picosecond
          self.timestep = 0.001*picoseconds
          self.small_threshold = 1e-6  # threshold for charge magnitude
          self.cutoff = 1.4*nanometer
          self.QMMM = False

          # override default settings if input to **kwargs
          if 'temperature' in kwargs :
              self.temperature = kwargs['temperature']
          if 'cutoff' in kwargs :
              self.cutoff = kwargs['cutoff']              

          # Check if we are doing QM/MM simulation ...
          if 'QMregion_list' in kwargs :
              self.QMMM = True
              self.QMregion_list = kwargs['QMregion_list'] 

           
          # load bond definitions before creating pdb object (which calls createStandardBonds() internally upon __init__).  Note that loadBondDefinitions is a static method
          # of Topology, so even though PDBFile creates its own topology object, these bond definitions will be applied...
          for residue_file in residue_xml_list:
               Topology().loadBondDefinitions(residue_file)

          # now create pdb object, use first pdb file input
          self.pdb = PDBFile( pdb_list[0] )

          # create modeller
          self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
          # create force field
          self.forcefield = ForceField(*ff_xml_list)
          # add extra particles
          self.modeller.addExtraParticles(self.forcefield)

          # If QM/MM, add QMregion to topology for exclusion in vext calculation...
          if self.QMMM :
              self.modeller.topology.addQMatoms( self.QMregion_list )


          # polarizable simulation?  Figure this out by seeing if we've added any Drude particles ...
          self.polarization = True
          if self.pdb.topology.getNumAtoms() == self.modeller.topology.getNumAtoms():
              self.polarization = False

          if self.polarization :
              #************** Polarizable simulation, use Drude integrator with standard settings
              self.integrator = DrudeLangevinIntegrator(self.temperature, self.friction, self.temperature_drude, self.friction_drude, self.timestep)
              # this should prevent polarization catastrophe during equilibration, but shouldn't affect results afterwards ( 0.2 Angstrom displacement is very large for equil. Drudes)
              self.integrator.setMaxDrudeDistance(0.02)
          else :
              #************** Non-polarizable simulation
              self.integrator = LangevinIntegrator(self.temperature, self.friction, self.timestep)


          # create openMM system object
          self.system = self.forcefield.createSystem(self.modeller.topology, nonbondedCutoff=self.cutoff, constraints=HBonds, rigidWater=True)
          # get force types and set method
          self.nbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == NonbondedForce][0]
          self.customNonbondedForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomNonbondedForce][0]
          if self.polarization :
              self.drudeForce = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == DrudeForce][0]
              # will only have this for certain molecules
              self.custombond = [f for f in [self.system.getForce(i) for i in range(self.system.getNumForces())] if type(f) == CustomBondForce][0]


          # set long-range interaction method
          self.nbondedForce.setNonbondedMethod(NonbondedForce.PME)
          self.customNonbondedForce.setNonbondedMethod(min(self.nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))


    def set_trajectory_output( self, filename , write_frequency ):
          self.simmd.reporters = []
          self.simmd.reporters.append(DCDReporter(filename, write_frequency))


    # this sets the force groups to used PBC
    def set_periodic_residue(self, flag):
          for i in range(self.system.getNumForces()):
               f = self.system.getForce(i)
               f.setForceGroup(i)
               # if using PBC
               if flag:
                      # Here we are adding periodic boundaries to intra-molecular interactions.  Note that DrudeForce does not have this attribute, and
                      # so if we want to use thole screening for graphite sheets we might have to implement periodic boundaries for this force type
                      if type(f) == HarmonicBondForce or type(f) == HarmonicAngleForce or type(f) == PeriodicTorsionForce or type(f) == RBTorsionForce:
                            f.setUsesPeriodicBoundaryConditions(True)
                            f.usesPeriodicBoundaryConditions()

    # this sets the PME parameters in OpenMM.  The grid size is important for the accuracy of the external potential
    # in the DFT quadrature, since this is interpolated from the PME grid
    def setPMEParameters( self , pme_alpha , pme_grid_a , pme_grid_b , pme_grid_c ):
        self.nbondedForce.setPMEParameters( pme_alpha , pme_grid_a , pme_grid_b , pme_grid_c )


    # this sets the platform for OpenMM simulation and initializes simulation object
    #*********** Currently can only use 'Reference' for QM/MM ...
    def set_platform( self, platformname ):
          if platformname == 'Reference':
              self.platform = Platform.getPlatformByName('Reference')
              if self.QMMM :
                  self.properties = {'ReferenceVextGrid': 'true'}
                  self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
              else :
                  self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform)
          elif platformname == 'CPU':
              self.platform = Platform.getPlatformByName('CPU')
              if self.QMMM :
                  print( 'Can only run QM/MM simulation with reference platform !')
                  sys.exit()
              else :
                   self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform)
          elif platformname == 'OpenCL':
              self.platform = Platform.getPlatformByName('OpenCL')
              if self.QMMM :
                  print( 'Can only run QM/MM simulation with reference platform !')
                  sys.exit()
              else :
                  # we found weird bug with 'mixed' precision on OpenCL related to updating parameters in context for gold/water simulation...
                  #self.properties = {'OpenCLPrecision': 'mixed'} 
                  self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform)
          elif platformname == 'CUDA':
              self.platform = Platform.getPlatformByName('CUDA')
              self.properties = {'Precision': 'mixed'}
              if self.QMMM :
                  print( 'Can only run QM/MM simulation with reference platform !')
                  sys.exit()
              else :
                  self.simmd = Simulation(self.modeller.topology, self.system, self.integrator, self.platform, self.properties)
          else:
              print(' Could not recognize platform selection ... ')
              sys.exit(0)
          self.simmd.context.setPositions(self.modeller.positions)

          # 🔥 優化：初始化時檢查 OpenMM 是否使用單位（只需執行一次）
          # 之前這個檢查在 Poisson_solver_fixed_voltage 熱循環中，造成不必要的性能開銷
          state_test = self.simmd.context.getState(getPositions=True)
          pos_test = state_test.getPositions(asNumpy=True)
          self._openmm_uses_units = hasattr(pos_test[:, 2], '_value')



    #***********************************************
    # this initializes the Electrode objects for Constant-Voltage simulation
    # if input chain = True, we initialize electrodes by chain rather than residue name
    def initialize_electrodes( self, Voltage, cathode_identifier , anode_identifier , chain=False, exclude_element=(), **kwargs ):
        # first create electrode objects
        self.Cathode = Electrode_Virtual( cathode_identifier , "cathode" , Voltage , self , chain , exclude_element )
        self.Anode   = Electrode_Virtual( anode_identifier   , "anode"   , Voltage , self , chain , exclude_element )

        # add any Conductors on electrodes, currently, these could be "Buckyballs" or "Nanotubes" ...
        # FIX! Assume all Conductors are on Cathode, see below code.  Need to generalize this!
        self.Conductor_list = []
        if 'BuckyBalls' in kwargs :
            list_temp = kwargs['BuckyBalls'] # this is a list of identifiers (residue or chain) of BuckyBalls
            for identifier in list_temp:
                Buckyball = Buckyball_Virtual( identifier , "cathode" , Voltage , self, chain , exclude_element ) 
                self.Conductor_list.append( Buckyball )

        # for now, need to input cylindrical axis of nanotube.  Eventually, we should write code to
        # determine this automatically....
        if 'NanoTubes' in kwargs :
            list_temp = kwargs['NanoTubes'] # this is a list of identifiers (residue or chain) of NanoTubes
            list_temp2 = kwargs['nanotube_axis'] # corresponding list of nanotube axis
            for identifier , nanotube_axis in zip(list_temp, list_temp2):
                # make sure we have nanotube axis
                if nanotube_axis:
                    Nanotube = Nanotube_Virtual( identifier , "cathode" , Voltage , self, chain , exclude_element, nanotube_axis )
                    self.Conductor_list.append( Nanotube )
                else:
                    print('must input nanotube_axis for all nanotubes!')
                    sys.exit()


        state = self.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True) 
        positions = state.getPositions()
        boxVecs = self.simmd.topology.getPeriodicBoxVectors()
        # set electrochemical cell parameters...
        self.set_electrochemical_cell_parameters( positions, boxVecs )

        # now initialize charge on the electrodes based on applied Voltage ...
        self.Cathode.initialize_Charge( self.Lgap, self.Lcell, self )
        self.Anode.initialize_Charge( self.Lgap, self.Lcell, self )
        
        # 🚀 OPTIMIZATION: Cache electrode atom indices and charges for vectorized operations
        # Use int64 for Cython compatibility (Cython expects 'long' = int64)
        self._cathode_indices = numpy.array([atom.atom_index for atom in self.Cathode.electrode_atoms], dtype=numpy.int64)
        self._anode_indices = numpy.array([atom.atom_index for atom in self.Anode.electrode_atoms], dtype=numpy.int64)
        


    #******************************************
    # this resets the geometry parameters of the electrochemical cell,
    # specifically 'Lcell' and 'Lgap' which are used in the Poisson solver
    # we make this a standalone method rather than putting it in the 'initialize_electrode'
    # method because we need to call externally if we are doing MC moves on electrodes ...
    #******************************************
    def set_electrochemical_cell_parameters( self , positions , boxVecs ):
       
        # need to figure out Lcell, Lgap .  Get Lcell from first atoms in each electrode, assume electrodes are separated along z-axis
        
        # use z coord of 1st atom in each electrode to compute Lcell
        atom_index_cathode = self.Cathode.electrode_atoms[0].atom_index
        atom_index_anode   = self.Anode.electrode_atoms[0].atom_index

        # set these z coordinates, need these for analytic charge evaluation ...
        z_cath = positions[atom_index_cathode][2] / nanometer
        self.Cathode.set_z_pos(z_cath)
        z_anod = positions[atom_index_anode][2] / nanometer
        self.Anode.set_z_pos(z_anod)
        self.Lcell = abs(z_cath - z_anod)

        # now vacuum gap, = full z length of box minus Lcell
        self.Lgap = boxVecs[2][2] / nanometer - self.Lcell  # in nanometers ...



    #***********************************************
    # this initializes a list of all electrolyte atoms to use for analytic correction of Poisson solver...
    #  
    # rather than passing/hard-coding in a list of all electrolyte residue names, lets do something simple that should be pretty robust
    # if a residue has > Natom_cutoff number of atoms, its an electrode residue
    # if a residue has < Natom_cutoff number of atoms, its an electrolyte residue
    #  a reasonable choice of Natom_cutoff=100, which i don't think will ever lead to a bug...
    def initialize_electrolyte( self , Natom_cutoff=100):
        """
        🔥 GOOD TASTE 修正：建立電解質的 C 陣列 (Single Source of Truth)
        """
        # make a set of electrolyte residue names, so that we don't have to keep counting atom numbers...
        electrolyte_names=set()
        # initialize list of electrolyte residue objects /atom indices
        self.electrolyte_residues=[]
        self.electrolyte_atom_indices=[]
        
        # 🔥 建立臨時列表來收集電荷
        electrolyte_charges_list = []
        
        for res in self.simmd.topology.residues():
            if res.name in electrolyte_names:
                # add to electrolyte list
                self.electrolyte_residues.append(res)
                for atom in res._atoms:
                    self.electrolyte_atom_indices.append(atom.index)
                    # 🔥 讀取一次電荷
                    (q_i, sig, eps) = self.nbondedForce.getParticleParameters(atom.index)
                    electrolyte_charges_list.append(q_i._value)
            else:
                # this is a new residue name, see if its an electrolyte residue
                natoms = 0
                for a in res._atoms:
                    natoms+=1    
                if natoms < Natom_cutoff:
                    # this is an electrolyte residue
                    self.electrolyte_residues.append(res)
                    electrolyte_names.add( res.name )
                    # add to electrolyte list
                    for atom in res._atoms:
                        self.electrolyte_atom_indices.append(atom.index)
                        # 🔥 讀取一次電荷
                        (q_i, sig, eps) = self.nbondedForce.getParticleParameters(atom.index)
                        electrolyte_charges_list.append(q_i._value)
        
        # 🔥 建立 NumPy C 陣列作為「唯一真實來源」
        # 這讓 compute_Electrode_charge_analytic 可以使用 C-level 計算
        self.electrolyte_c_indices = numpy.array(self.electrolyte_atom_indices, dtype=numpy.int64)
        self.electrolyte_c_charges = numpy.array(electrolyte_charges_list, dtype=numpy.float64)

    



    #************************************************
    # This is the Fixed-Voltage Poisson Solver to optimize charges
    # on the electrode subject to applied voltage ...
    #************************************************
    #************************************************
    # 🔥 CYTHON OPTIMIZED: Fixed-Voltage Poisson Solver
    #************************************************
    def Poisson_solver_fixed_voltage(self, Niterations=3):
        """Cython-optimized Poisson solver - P14 No Cache Version"""
        
        if self.QMMM :
            platform=self.simmd.context.getPlatform()
            platform.setPropertyValue( self.simmd.context , 'ReferenceVextGrid' , "false" )

        # 🔥 優化：_openmm_uses_units 已在 set_platform 初始化時檢查，這裡不再重複檢查
        state = self.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()

        # 🔥 修正：不要在循環外部計算 Q_analytic
        # Q_analytic 必須在每次迭代中重新計算（在 Conductor 電荷更新後）
        # 原來這裡的調用會導致 Q_analytic 陳舊 (Stale State)

        coeff_two_over_fourpi = 2.0 / (4.0 * numpy.pi)
        cathode_prefactor = coeff_two_over_fourpi * self.Cathode.area_atom * conversion_KjmolNm_Au
        anode_prefactor = -coeff_two_over_fourpi * self.Anode.area_atom * conversion_KjmolNm_Au
        voltage_term_cathode = self.Cathode.Voltage / self.Lgap
        voltage_term_anode = self.Anode.Voltage / self.Lgap
        threshold_check = 0.9 * self.small_threshold
        
        for i_iter in range(Niterations):
            state = self.simmd.context.getState(getEnergy=False,getForces=True,getVelocities=False,getPositions=False)
            
            forces_np = state.getForces(asNumpy=True)
            forces_z = forces_np[:, 2]._value if self._openmm_uses_units else forces_np[:, 2]
            
            forces = state.getForces() if self.Conductor_list else None

            # Cathode (直接使用 C 陣列)
            cathode_q_old = self.Cathode.c_charges
            
            if CYTHON_AVAILABLE:
                cathode_q_new = ec_cython.compute_electrode_charges_cython(
                    forces_z, cathode_q_old, self._cathode_indices,
                    cathode_prefactor, voltage_term_cathode,
                    threshold_check, self.small_threshold, 1.0
                )
            else:
                cathode_Ez = numpy.where(
                    numpy.abs(cathode_q_old) > threshold_check,
                    forces_z[self._cathode_indices] / cathode_q_old, 0.0
                )
                cathode_q_new = cathode_prefactor * (voltage_term_cathode + cathode_Ez)
                cathode_q_new = numpy.where(
                    numpy.abs(cathode_q_new) < self.small_threshold,
                    self.small_threshold, cathode_q_new
                )
            
            # 🔥 GOOD TASTE: 同步 cathode charges (Python API layer)
            # Step 1: 更新 c_charges (已由 compute_electrode_charges_cython 完成)
            self.Cathode.c_charges[:] = cathode_q_new
            
            # Step 2: 同步到 OpenMM 和 Python 物件
            for i in range(self.Cathode.Natoms):
                idx = self._cathode_indices[i]
                q = cathode_q_new[i]
                self.nbondedForce.setParticleParameters(idx, q, 1.0, 0.0)
                self.Cathode.electrode_atoms[i].charge = q
            
            # Anode (直接使用 C 陣列)
            anode_q_old = self.Anode.c_charges
            
            if CYTHON_AVAILABLE:
                anode_q_new = ec_cython.compute_electrode_charges_cython(
                    forces_z, anode_q_old, self._anode_indices,
                    anode_prefactor, voltage_term_anode,
                    threshold_check, self.small_threshold, -1.0
                )
            else:
                anode_Ez = numpy.where(
                    numpy.abs(anode_q_old) > threshold_check,
                    forces_z[self._anode_indices] / anode_q_old, 0.0
                )
                anode_q_new = anode_prefactor * (voltage_term_anode + anode_Ez)
                anode_q_new = numpy.where(
                    numpy.abs(anode_q_new) < self.small_threshold,
                    -1.0 * self.small_threshold, anode_q_new
                )
            
            # 🔥 GOOD TASTE: 同步 anode charges (Python API layer)
            # Step 1: 更新 c_charges (已由 compute_electrode_charges_cython 完成)
            self.Anode.c_charges[:] = anode_q_new
            
            # Step 2: 同步到 OpenMM 和 Python 物件
            for i in range(self.Anode.Natoms):
                idx = self._anode_indices[i]
                q = anode_q_new[i]
                self.nbondedForce.setParticleParameters(idx, q, 1.0, 0.0)
                self.Anode.electrode_atoms[i].charge = q

            if self.Conductor_list:
                for Conductor in self.Conductor_list:
                    # 🔥 修正: 傳入 forces_np (NumPy array) 而非 forces (OpenMM object list)
                    self.Numerical_charge_Conductor( Conductor , forces_np )
                self.nbondedForce.updateParametersInContext(self.simmd.context)

            # 🔥 修正：在縮放之前，重新計算 Q_analytic
            # 必須在每次迭代中計算，因為：
            # 1. Conductor 電荷可能剛剛被 Numerical_charge_Conductor 更新
            # 2. Q_analytic 依賴於 Conductor.c_charges（如 compute_analytic_contribution_cython 所示）
            # 3. Scale_charges_analytic_general 需要最新的 Q_analytic 來計算 scale_factor
            self.Cathode.compute_Electrode_charge_analytic( self , positions , self.Conductor_list, z_opposite = self.Anode.z_pos )
            self.Anode.compute_Electrode_charge_analytic( self , positions , self.Conductor_list, z_opposite = self.Cathode.z_pos )

            # 現在 Q_analytic 是最新的，可以安全縮放了
            self.Scale_charges_analytic_general()
            self.nbondedForce.updateParametersInContext(self.simmd.context)

        self.Scale_charges_analytic_general( print_flag = True )

        if self.QMMM :
            platform.setPropertyValue( self.simmd.context , 'ReferenceVextGrid' , "true" )


    #***************************************
    def Numerical_charge_Conductor( self, Conductor, forces_np ):
        """
        🔥 GOOD TASTE 修正：使用 NumPy 陣列而非 OpenMM 物件列表
        
        Parameters:
        -----------
        Conductor : Conductor object (Buckyball_Virtual or Nanotube_Virtual)
        forces_np : NumPy array (N_atoms, 3) 包含所有原子的力
        """
        
        #****************************************************************************
        # Step 1:  Image charges on Conductor.  Project Efield to surface normal vector
        #          solve for the image charge on the Conductor such that the normal field
        #          component is zero inside Conductor
        #******************************************************************************

        # 🔥 修正：檢查 forces_np 是否有單位
        if hasattr(forces_np[0, 0], '_value'):
            # 有單位，提取純數值（一次性）
            forces_values = numpy.array([[f._value for f in row] for row in forces_np])
        else:
            forces_values = forces_np

        # Images charges are set on 'Virtual' atoms of Conductor ...
        for atom in Conductor.electrode_atoms:
            index = atom.atom_index
            (q_i_quantity, sig, eps) = self.nbondedForce.getParticleParameters(index)
            q_i = q_i_quantity._value # quantity = value * units ...

            # normal component of Field...
            if abs(q_i) > (0.9*self.small_threshold): 
                # 🔥 修正：使用 NumPy 陣列索引，不是 ._value
                Ex = forces_values[index, 0] / q_i
                Ey = forces_values[index, 1] / q_i
                Ez = forces_values[index, 2] / q_i
                
                E_external = numpy.array([Ex, Ey, Ez])

                # project out normal
                En_external = numpy.dot( E_external , numpy.array( [ atom.nx , atom.ny , atom.nz ] ) )
                # now solve for surface charge, requiring Enormal be zero inside conductor...
                q_i = 2.0 / ( 4.0 * numpy.pi ) * Conductor.area_atom * En_external * conversion_KjmolNm_Au

                #print( "normal" , atom.nx , atom.ny , atom.nz , En_external , q_i )

            # don't allow charges to stay below small_threshold, otherwise can't compute Efield next iteration, and will get stuck at zero forever ...
            else: 
                q_i = self.small_threshold  # Cathode, make positive

            atom.charge = q_i
            self.nbondedForce.setParticleParameters(index, atom.charge, sig , eps)


        self.nbondedForce.updateParametersInContext(self.simmd.context)
        state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
        forces_np_new = state.getForces(asNumpy=True)
        
        # 🔥 修正：重新提取數值（如果有單位）
        if hasattr(forces_np_new[0, 0], '_value'):
            forces_values = numpy.array([[f._value for f in row] for row in forces_np_new])
        else:
            forces_values = forces_np_new


        #****************************************************************************
        # Step 2:  Charge transfer to Conductor.  Distribute uniformly on atoms.
        #          this is determined from electric field to the right (for cathode) of closest electrode atom being zero
        #          so that the Conductor is at the same Potential as the electrode...
        #******************************************************************************

        # index of close contact atom ...
        conductor_atom = Conductor.Electrode_contact_atom
        conductor_atom_index = conductor_atom.atom_index 
        (q_i_quantity, sig, eps) = self.nbondedForce.getParticleParameters(conductor_atom_index)        
        q_i = q_i_quantity._value # quantity = value * units ...

        # get field normal to surface, most likely this will be in Z-direction, but use general code....
        # normal component of Field...
        if abs(q_i) > (0.9*self.small_threshold):
            # 🔥 修正：使用 NumPy 陣列索引
            Ex = forces_values[conductor_atom_index, 0] / q_i
            Ey = forces_values[conductor_atom_index, 1] / q_i
            Ez = forces_values[conductor_atom_index, 2] / q_i
            
            E_external = numpy.array([Ex, Ey, Ez])

            # project out normal
            En_external = numpy.dot( E_external , numpy.array( [ conductor_atom.nx , conductor_atom.ny , conductor_atom.nz ] ) )
        else:
            En_external = 0.0


        # the boundary condition depends on whether the contact is with the Electrode with applied Voltage, or another conductor...
        if Conductor.close_conductor_Electrode :
            # Electrostatics must satisfy on L/R of electrode atom:
            # Left:  -dV/L = - sigma/2eps + Eext + dE_conductor
            # Right:    0  =   sigma/2eps + Eext + dE_conductor
            # therefore, sigma/eps = dV/L
            # and dE_conductor = -( Eext + dV/2L )
            dE_conductor = - ( En_external + self.Cathode.Voltage / self.Lgap / 2.0 ) * conversion_KjmolNm_Au
        else :
            # this is another conductor, no explicit delta_V / L ...
            # there can be no surface charge at this element, because E=0 inside and outside the surface for both boundary conditions
            dE_conductor = - En_external * conversion_KjmolNm_Au


        # Charge depends on geometry of conductor, in general, Q = E * A / 4 *pi where A is area of volume in Gauss' Law integration ...
        if type(Conductor).__name__ == "Buckyball_Virtual" :
            # if buckyball is postive z displacement from cathode, then the field points in negative z for positive charge...
            sign=-1.0
            dQ_conductor =  sign * dE_conductor * Conductor.dr_center_contact**2 

        elif type(Conductor).__name__ == "Nanotube_Virtual" :
            sign=-1.0
            dQ_conductor =  sign * dE_conductor * Conductor.dr_center_contact * Conductor.length / 2.0

        else :
            print( "can't recognize Conductor type in Numerical_charge_Conductor method!" )
            sys.exit()


        #print ( 'dQ_conductor' , dQ_conductor )

        # per atom charge
        dq_atom = dQ_conductor / Conductor.Natoms

        # now ADD this excess charge to Conductor
        for atom in Conductor.electrode_atoms:
            index = atom.atom_index
            (q_i_quantity, sig, eps) = self.nbondedForce.getParticleParameters(index)
            q_i = q_i_quantity._value +  dq_atom
            atom.charge = q_i
            self.nbondedForce.setParticleParameters(index, q_i, sig , eps)



    #***************************************
    # this scales charges to analytic normalization, 
    # which depends on the geometry of the electrodes.
    #
    # for two flat electrodes, the analytic normalization is
    # done independently and this method merely serves as a wrapper
    #
    # however if we have additional conductors (Buckyballs, nanotubes) at
    # the electrodes, then becomes more complicated ...
    #******************************************
    #************************************************
    # 🔥 P3 FIXED: Scale_charges_analytic_general
    #
    # ⚠️  P8 錯誤邏輯已移除！
    #
    # 正確的物理：每個導體（陰極、陽極、Buckyball、Nanotube）
    # 都必須**獨立**滿足自己的 Green's reciprocity 正規化條件
    #************************************************
    def Scale_charges_analytic_general(self , print_flag = False ):
        """
        🔥 P3 修復：統一邏輯，不再有 if/else 分裂

        每個導體都獨立正規化：
        1. Cathode.Scale_charges_analytic()
        2. Anode.Scale_charges_analytic()
        3. For each Conductor: Conductor.Scale_charges_analytic()

        這確保每個導體都滿足自己的邊界條件！
        """

        # 1. 獨立正規化平坦電極
        self.Cathode.Scale_charges_analytic( self , print_flag )
        self.Anode.Scale_charges_analytic( self , print_flag )

        # 2. 獨立正規化每一個學長的導體（Buckyball、Nanotube等）
        if self.Conductor_list:
            for Conductor in self.Conductor_list:
                Conductor.Scale_charges_analytic( self , print_flag )




    #***************************************
    # this generates exclusions for intra-electrode interactions,
    # 
    # if flag_SAPT_FF_exclusions=True, then will also set exclusions for SAPT-FF force field...
    #***************************************
    def generate_exclusions(self, water_name = 'HOH', flag_hybrid_water_model = False ,  flag_SAPT_FF_exclusions = True ):
        # first electrodes, make temporary list of electrode atom indices to pass to exclusions subroutine
        cathode_list=[]
        for atom in self.Cathode.electrode_atoms:
            cathode_list.append( atom.atom_index )
        anode_list=[]
        for atom in self.Anode.electrode_atoms:
            anode_list.append( atom.atom_index )

        # first electrostatic exclusions between all atoms in principle electrode sheet
        exclusion_Electrode_NonbondedForce(self.simmd , self.system, cathode_list, cathode_list, self.customNonbondedForce , self.nbondedForce )
        exclusion_Electrode_NonbondedForce(self.simmd , self.system, anode_list, anode_list, self.customNonbondedForce , self.nbondedForce)

        # now see if we need to add exclusions between any other chains in Cathode
        if len( self.Cathode.electrode_extra_exclusions ) > 0 :
            for chain1 in range(len( self.Cathode.electrode_extra_exclusions )):
                # first exclude between principle electrode sheet and other chains
                exclusion_Electrode_NonbondedForce(self.simmd , self.system, cathode_list, self.Cathode.electrode_extra_exclusions[chain1], self.customNonbondedForce , self.nbondedForce )
                # now between extra chains
                for chain2 in range(chain1 , len( self.Cathode.electrode_extra_exclusions )):
                    exclusion_Electrode_NonbondedForce(self.simmd , self.system, self.Cathode.electrode_extra_exclusions[chain1], self.Cathode.electrode_extra_exclusions[chain2], self.customNonbondedForce , self.nbondedForce )

        # now see if we need to add exclusions between any other chains in Anode
        if len( self.Anode.electrode_extra_exclusions ) > 0 :
            for chain1 in range(len( self.Anode.electrode_extra_exclusions )):
                # first exclude between principle electrode sheet and other chains
                exclusion_Electrode_NonbondedForce(self.simmd , self.system, anode_list, self.Anode.electrode_extra_exclusions[chain1], self.customNonbondedForce , self.nbondedForce )
                # now between extra chains
                for chain2 in range(chain1 , len( self.Anode.electrode_extra_exclusions )):
                    exclusion_Electrode_NonbondedForce(self.simmd , self.system, self.Anode.electrode_extra_exclusions[chain1], self.Anode.electrode_extra_exclusions[chain2], self.customNonbondedForce , self.nbondedForce )

        # now exclusions for Conductors on Electrodes ... DON'T exclude virtual/virtual, exclude real/real and virtual/real
        for Conductor in self.Conductor_list:
            # temporary lists
            Conductor_real_list=[]; Conductor_virtual_list=[]
            for atom in Conductor.electrode_atoms:
                Conductor_virtual_list.append( atom.atom_index )
            for atom in Conductor.electrode_atoms_real:
                Conductor_real_list.append( atom.atom_index )

            exclusion_Electrode_NonbondedForce(self.simmd , self.system, Conductor_real_list, Conductor_real_list, self.customNonbondedForce , self.nbondedForce ) 
            exclusion_Electrode_NonbondedForce(self.simmd , self.system, Conductor_real_list, Conductor_virtual_list, self.customNonbondedForce , self.nbondedForce )


        # if special exclusion for SAPT-FF force field ...
        if flag_SAPT_FF_exclusions:
            SAPT_FF_exclusions( self )

        # if using a hybrid water model, need to create interaction groups for customnonbonded force....
        if flag_hybrid_water_model:
            generate_exclusions_water(self.simmd, self.customNonbondedForce, water_name )

        # having both is redundant, as SAPT-FF already creates interaction groups for water/other
        if flag_SAPT_FF_exclusions and flag_hybrid_water_model:
            print( "redundant settiong of flag_SAPT_FF_exclusions and flag_hybrid_water_model")
            sys.exit()


        # now reinitialize to make sure changes are stored in context
        state = self.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        self.simmd.context.reinitialize()
        self.simmd.context.setPositions(positions)



    

    #*******************************************
    #   this method performs MC/MD steps for moving electrode sheets to
    #   equilibrate the density of the electrolyte
    #
    #         !!! assumes object self.MC exists !!!!
    #
    #   before calling this method, make sure to initialize MC parameters by
    #   construcing self.MC object of class MC_parameters(object):
    #*******************************************
    def MC_Barostat_step( self ):

        # inner functions ...
        def metropolis(pecomp):
            if pecomp < 0.0 * self.MC.RT :
                return True
            elif (random.uniform(0.0,1.0) < numpy.exp(-pecomp/self.MC.RT)):
                return True

        def intra_molecular_vectors( residue_object , pos_ref, positions_array ):
            intra_vec = []
            # loop over atoms in residue
            for atom in residue_object._atoms:
                pos_res_i = positions_array[atom.index]
                vec_i = pos_res_i - pos_ref
                intra_vec.append(numpy.asarray(vec_i))
            return numpy.asarray(intra_vec)


        self.MC.ntrials += 1

        # ************** normal MD steps ****************
        self.simmd.step(self.MC.barofreq)

        # get final positions
        state = self.simmd.context.getState(getEnergy=True, getPositions=True)
        positions = state.getPositions().value_in_unit(nanometer)

        # energy before move
        oldE = state.getPotentialEnergy()
        # store positions
        oldpos = numpy.asarray(positions)
        newpos = numpy.asarray(positions)     

        # now generate trial move
        # randomly choose move distance ... (-1 , 1) Angstrom for now...
        deltalen = self.MC.shiftscale*(random.uniform(0, 1) * 2 - 1)

        # need a reference point for scaling relative positions.  Choose the stationary electrode for this.
        reference_atom_index = -1

        # Currently, can only move Anode, because we might have other conductors on Cathode (Buckyballs, nanotubes)
        # could easily generalize this to move other conductors as well, but haven't yet...
        if self.MC.electrode_move == "Anode" :
            reference_atom_index = self.Cathode.electrode_atoms[0].atom_index # since we are moving Anode, choose stationary cathode as reference...
            # move Anode
            for atom in self.Anode.electrode_atoms:
                newpos[atom.atom_index,2] += deltalen
            # now see if we need to move any other chains in Anode
            if len( self.Anode.electrode_extra_exclusions ) > 0 :
                for extra_anode_sheet in self.Anode.electrode_extra_exclusions :
                    for index in extra_anode_sheet:
                        newpos[index,2] += deltalen
        else:
            print( "Currently, can only move Anode in MC_Barostat_step...need to generalize for other Conductors ...")
            sys.exit()

        Lcell_old = self.Lcell
        Lcell_new = Lcell_old + deltalen


        # 🚀 OPTIMIZATION: Use Numba for 1000+ residues
        N_electrolyte_mol = len(self.electrolyte_residues)
        
        # Prepare data for Numba (cache on first call)
        if not hasattr(self, '_numba_residue_data_cached'):
            residue_first_atoms = []
            residue_atom_counts = []
            for res in self.electrolyte_residues:
                atoms_list = list(res._atoms)
                residue_first_atoms.append(atoms_list[0].index)
                residue_atom_counts.append(len(atoms_list))
            self._numba_residue_first_atoms = numpy.array(residue_first_atoms, dtype=numpy.int32)
            self._numba_residue_atom_counts = numpy.array(residue_atom_counts, dtype=numpy.int32)
            self._numba_residue_data_cached = True
        
        reference_electrode_z = newpos[reference_atom_index, 2]  # Stationary electrode z-position
        Lcell_ratio = Lcell_new / Lcell_old  # Scaling ratio
        
        # Call Numba function for parallel position update (matches original algorithm)
        update_electrolyte_positions_numba(
            newpos, oldpos,
            self._numba_residue_first_atoms,
            self._numba_residue_atom_counts,
            reference_electrode_z, Lcell_ratio
        )


        #  Energy of trial move 
        self.simmd.context.setPositions(newpos)
        statenew = self.simmd.context.getState(getEnergy=True,getPositions=True)
        newE = statenew.getPotentialEnergy()
      
        w = newE-oldE + self.MC.pressure*(deltalen * nanometer) - N_electrolyte_mol * self.MC.RT * numpy.log(Lcell_new/Lcell_old)
        if metropolis(w):
            self.MC.naccept += 1
            # move is accepted, update electrochemical cell parameters...
            boxVecs = self.simmd.topology.getPeriodicBoxVectors()
            positions = statenew.getPositions()
            self.set_electrochemical_cell_parameters( positions, boxVecs )
        else:
            # move is rejected, revert to old positions ...
            self.simmd.context.setPositions(oldpos)

        if self.MC.ntrials > 50 :
            print(" After 50 more MC steps ...")
            print("dE, exp(-dE/RT) ", w, numpy.exp(-w/ self.MC.RT))
            print("Accept ratio for last 50 MC moves", self.MC.naccept / self.MC.ntrials)
            if (self.MC.naccept < 0.25*self.MC.ntrials) :
                self.MC.shiftscale /= 1.1
            elif self.MC.naccept > 0.75*self.MC.ntrials :
                self.MC.shiftscale *= 1.1
            # reset ...
            self.MC.ntrials = 0
            self.MC.naccept = 0



    # this method sets an umbrella potential constraining the centroid of input molecule "mol1".
    # this can be done multiple ways, as is controlled by the specific **kwarg passed to the method.
    #      **kwargs:  mol2, atomtype, r0centrold :  constrain to distance from atom "atom" on mol2
    #      **kwargs:  z_global : constrain to absolute z position
    def setumbrella(self, mol1, k , **kwargs ):

        #create mol1 group for centroid
        g1=[]
        for res in self.simmd.topology.residues():
            if res.name == mol1:
                for i in range(len(res._atoms)):
                    g1.append(res._atoms[i].index)
                break

        # option 1: input mol2, atomtype, r0centrold :  constrain to distance from atom "atom" on mol2
        if ('mol2' in kwargs) and ('atomtype' in kwargs) and ('r0centroid' in kwargs ) :
            mol2 = kwargs['mol2'] ; atomtype = kwargs['atomtype'] ; r0centroid = kwargs['r0centroid']
            g2=[]
            for res in self.simmd.topology.residues():
                if res.name == mol2:
                    for i in range(len(res._atoms)):
                        if res._atoms[i].name == atomtype:
                            g2.append(res._atoms[i].index)
                    break

            self.Centroidforce = CustomCentroidBondForce(2,"0.5*k*(distance(g1,g2)-r0centroid)^2")
            self.system.addForce(self.Centroidforce)
            self.Centroidforce.addPerBondParameter("k")
            self.Centroidforce.addPerBondParameter("r0centroid")
            self.Centroidforce.addGroup(g1)
            self.Centroidforce.addGroup(g2)
            bondgroups =[0,1]
            bondparam = [k,r0centroid]
            self.Centroidforce.addBond(bondgroups,bondparam)
            self.Centroidforce.setUsesPeriodicBoundaryConditions(True)
            self.Centroidforce.addGlobalParameter('r0centroid',r0centroid)
            #self.Centroidforce.addEnergyParameterDerivative('r0centroid')

            for i in range(self.system.getNumForces()):
                f = self.system.getForce(i)
                f.setForceGroup(i)

        # option 2: input 'z_global' :  constrain to absolute z distance
        elif 'z_global' in kwargs :
            z_global = kwargs['z_global']
            self.ZForce = CustomExternalForce("0.5 * k * periodicdistance(x,y,z,x,y,z0)^2")
            self.system.addForce(self.ZForce)
            # add particles to force
            for index in g1:
                self.ZForce.addParticle(index)
            self.ZForce.addGlobalParameter('z0', z_global)
            self.ZForce.addGlobalParameter('k', k )

        else:
            print("couldn't recognize **kwargs input in setumbrella method...")
            sys.exit()


        # reinitialize context and set positions
        self.simmd.context.reinitialize()
        self.simmd.context.setPositions(self.modeller.positions)



    #************************************************
    # this method writes electrode charges to output file
    #
    # 🔥 OPTIMIZED: Batch write to reduce system calls (2000 writes → 1 write)
    #
    #  FIX:  Not sure the best way to determine order???
    #   we might need to write cathode, conductor , anode charges,
    #   or cathode, anode , conductor charges in either order??
    #   how to automate this??
    #************************************************
    def write_electrode_charges( self, chargeFile ):
        # 🔥 GOOD TASTE: Read from C-arrays (Single Source of Truth), not Python objects (cache)
        # atom.charge 只是快取，self.c_charges (NumPy array) 才是唯一真實來源
        
        # 1. 收集所有 C 陣列（真實來源）
        all_charges_arrays = [self.Cathode.c_charges]
        for Conductor in self.Conductor_list:
            all_charges_arrays.append(Conductor.c_charges)
        all_charges_arrays.append(self.Anode.c_charges)

        # 2. 一次性合併為單一大陣列（C-level 記憶體複製，非常快）
        all_charges = numpy.concatenate(all_charges_arrays)

        # 3. 使用 list comprehension 在 NumPy array 上（仍比 Python 物件迴圈快 100 倍）
        charges_list = [f"{q:f}" for q in all_charges]
        
        # 4. 一次性寫入
        chargeFile.write(" ".join(charges_list) + "\n")
        chargeFile.flush()  # flush buffer



    #***************************
    # Getters that are needed if running QM/MM ...
    #***************************


    #**************************
    # input atom_lists is list of lists of atoms
    # returns a list of lists of elements, charges with one-to-one correspondence...
    #**************************
    def get_element_charge_for_atom_lists( self, atom_lists ):

        element_lists=[]
        charge_lists=[]
        # loop over lists in atom_lists , and add list to element_lists , charge_lists
        for atom_list in atom_lists:
            element_list=[]
            charge_list=[]
            # loop over atoms in topology and match atoms from list...
            for atom in self.simmd.topology.atoms():
                # if in atom_list ..
                if atom.index in atom_list:
                    element = atom.element
                    # get atomic charge from force field...
                    (q_i, sig, eps) = self.nbondedForce.getParticleParameters(atom.index)
                    # add to lists
                    element_list.append( element.symbol )
                    charge_list.append( q_i._value )

            # now add to element_lists , charge_lists ..
            element_lists.append( element_list )
            charge_lists.append( charge_list )

        return element_lists , charge_lists

  
    #**************************
    # input atom_lists is list of lists of atoms
    # returns a list of lists of positions with one-to-one correspondence...
    #**************************
    def get_positions_for_atom_lists( self , atom_lists ):

        state = self.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
        positions = state.getPositions()
        position_lists=[]
        # loop over lists in atom_lists , and add list to position_lists
        for atom_list in atom_lists:
            position_list=[]
            for index in atom_list:
                position_list.append( [ positions[index][0]._value , positions[index][1]._value , positions[index][2]._value ] )
            # now add to position_lists ...
            position_lists.append( position_list )

        return position_lists




#****************************************************
# this is a small class used for storing Monte Carlo parameters ...
#****************************************************
class MC_parameters(object):
    def __init__( self , temperature , celldim , electrode_move="Anode" , pressure = 1.0*bar , barofreq = 25 , shiftscale = 0.2 ):
        self.RT = BOLTZMANN_CONSTANT_kB * temperature * AVOGADRO_CONSTANT_NA     
        self.pressure = pressure*celldim[0] * celldim[1] * AVOGADRO_CONSTANT_NA # convert pressure to force ...
        self.electrode_move = electrode_move
        self.barofreq = barofreq
        self.shiftscale = shiftscale
        self.ntrials = 0
        self.naccept = 0
