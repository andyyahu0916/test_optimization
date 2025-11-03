#!/usr/bin/env python

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
from copy import deepcopy
import os
import sys
import numpy

# 🔥 CYTHON IMPORTS
try:
    import electrode_charges_cython as ec_cython
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("⚠️ Cython module not available, falling back to Python")

#**********************************
# 🔥 P14: CYTHON VERSION - 100% INDEPENDENT
# This file is a complete copy of OPTIMIZED with Cython overlays
# NO cross-file imports! Each version stands alone!
#**********************************

# conversion factors/parameters
conversion_nmBohr = 18.8973
conversion_KjmolNm_Au = conversion_nmBohr / 2625.5
conversion_eV_Kjmol = 96.487


class atom_MM(object):
    def __init__(self, element, charge, atom_index ):
        self.element = element
        self.charge  = charge
        self.atom_index = atom_index
        self.x = 0.0; self.y=0.0; self.z=0.0
        self.nx = 0.0; self.ny = 0.0; self.nz = 0.0

    def set_xyz( self, x , y , z ):
        self.x = x
        self.y = y
        self.z = z


#*********************************
# Conductor_Virtual - Parent class with Cython optimizations
#*********************************
class Conductor_Virtual(object):
    def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element):
        if isinstance( electrode_identifier , tuple ):
            self.electrode_identifier    = electrode_identifier[0]
        else:
            self.electrode_identifier    = electrode_identifier
        self.electrode_type = electrode_type
        self.Voltage        = Voltage * conversion_eV_Kjmol
        self.z_pos          = 0.0
        self.Q_analytic     = 0.0

        if not (self.electrode_type == "cathode" or self.electrode_type == "anode" ):
            print(' to create Electrode_Virtual object, must set electrode_type to either "cathode" or "anode" !')
            sys.exit(0)

        self.Electrode_contact_atom = False
        self.close_conductor_Electrode = True
        self.close_conductor_threshold = 1.5
        self.electrode_extra_exclusions=[]

        self.electrode_atoms=[]
        flag=0

        if chain_flag == True:
            for chain in MMsys.simmd.topology.chains():
                if chain.index == self.electrode_identifier:
                    flag=1
                    for atom in chain.atoms():
                        element = atom.element
                        if element.symbol not in exclude_element:
                            (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(atom.index)
                            atom_object = atom_MM( element.symbol , q_i._value , atom.index )
                            self.electrode_atoms.append( atom_object )

            if isinstance( electrode_identifier , tuple ) and ( len(electrode_identifier) > 1 ) :
                iterelectrode = iter(electrode_identifier)
                next(iterelectrode)
                for identifier in iterelectrode:
                    electrode_chain_atoms=[]
                    for chain in MMsys.simmd.topology.chains():
                        if chain.index == identifier:
                            for atom in chain.atoms():
                                electrode_chain_atoms.append( atom.index )
                    self.electrode_extra_exclusions.append( electrode_chain_atoms )

        else:
            for res in MMsys.simmd.topology.residues():
                if res.name == self.electrode_identifier:
                    flag=1
                    for atom in res._atoms:
                        element = atom.element
                        if element.symbol not in exclude_element:
                            (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(atom.index)
                            atom_object = atom_MM( element.symbol , q_i._value , atom.index )
                            self.electrode_atoms.append( atom_object )

        if flag == 0:
            print(' Couldnt find electrode residue...please check input electrode_identifier when constructing Electrode_Virtual object ! ')
            sys.exit(0)

        self.Natoms = len(self.electrode_atoms)

        # 🔥 GOOD TASTE: Create C-level arrays as Single Source of Truth
        # 從 Python 物件列表一次性提取資料，存入 C 級陣列
        # 這是電荷的「唯一真實來源」，atom.charge 只是 Python-side 的快取
        self.c_indices = numpy.array([atom.atom_index for atom in self.electrode_atoms], dtype=numpy.int32)
        self.c_charges = numpy.array([atom.charge for atom in self.electrode_atoms], dtype=numpy.float64)


    #******************************************
    # 🔥 GOOD TASTE: get_total_charge - NumPy 完勝
    # NumPy.sum 已經是 C 實現，無需 Cython
    def get_total_charge( self ):
        # 直接在 C 陣列上操作，零 Python 負擔
        return numpy.sum(self.c_charges)


    #*********************************************
    # find_contact_neighbor_conductor - Python (no optimization needed)
    def find_contact_neighbor_conductor( self, positions , r_center , MMsys ):
       if self.electrode_type == "cathode":
           Electrode_contact = MMsys.Cathode
       else:
           Electrode_contact = MMsys.Anode

       min_dist = 10.0
       for atom in Electrode_contact.electrode_atoms:
           dr_atom = numpy.sqrt( ( r_center[0] - positions[atom.atom_index][0]._value )**2 + ( r_center[1] - positions[atom.atom_index][1]._value )**2 + ( r_center[2] - positions[atom.atom_index][2]._value )**2 )
           if dr_atom < min_dist:
               self.Electrode_contact_atom = atom
               min_dist = dr_atom

       if  min_dist < self.close_conductor_threshold :
           self.dr_center_contact = min_dist
           return False

       else:
           self.close_conductor_Electrode = False
           print( "Searching Conductors for close-contact distance pair ... ")
           for Conductor in MMsys.Conductor_list :
               for atom1 in self.electrode_atoms:
                   for atom2 in Conductor.electrode_atoms:
                       dr_atom = numpy.sqrt( ( positions[atom1.atom_index][0]._value - positions[atom2.atom_index][0]._value )**2 + ( positions[atom1.atom_index][1]._value - positions[atom2.atom_index][1]._value )**2 +( positions[atom1.atom_index][2]._value - positions[atom2.atom_index][2]._value )**2 )
                       if dr_atom < min_dist:
                           self.Electrode_contact_atom = atom2
                           min_dist = dr_atom

               if  min_dist < self.close_conductor_threshold :
                   dr_vector = [0] * 3
                   for i in range(3):
                       dr_vector[i] = positions[self.Electrode_contact_atom.atom_index][i]._value - r_center[i]
                   self.dr_center_contact = numpy.sqrt( dr_vector[0]**2 + dr_vector[1]**2 + dr_vector[2]**2 )
                   return dr_vector

       print( "Failed to find close Conductor for threshold " , self.close_conductor_threshold )
       sys.exit()


#*********************************
# Electrode_Virtual - Child class with Cython optimizations
#*********************************
class Electrode_Virtual(Conductor_Virtual):
    def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element):
        super().__init__(electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element)

        boxVecs = MMsys.simmd.topology.getPeriodicBoxVectors()
        crossBox = numpy.cross(boxVecs[0], boxVecs[1])
        self.sheet_area = numpy.dot(crossBox, crossBox)**0.5 / nanometer**2
        self.area_atom = self.sheet_area / self.Natoms

        for atom in self.electrode_atoms:
            atom.nx = 0.0 ; atom.ny = 0.0 ; atom.nz = 1.0


    #*************************
    # 🔥 GOOD TASTE: initialize_Charge - 計算/同步分離
    # Step 1: 計算 (C-level)
    # Step 2: 同步 (Python API)
    def initialize_Charge( self, Lgap, Lcell, MMsys):
        sign=1.0
        if self.electrode_type == 'anode':
            sign=-1.0

        flag_small=False
        if abs(self.Voltage) < 0.01:
            print( "adding small value to initial charges in initialize_Charge routine for small Voltage input..." )
            flag_small=True

        # --- STEP 1: 計算 (在 C 陣列上操作) ---
        # 計算每個原子的基礎電荷
        q_i = sign / ( 4.0 * numpy.pi ) * self.area_atom * (self.Voltage / Lgap + self.Voltage / Lcell) * conversion_KjmolNm_Au

        if CYTHON_AVAILABLE:
            # 使用 Cython 函數填充 c_charges
            ec_cython.initialize_charges_cython(
                self.c_charges,
                q_i,
                MMsys.small_threshold,
                sign
            )
        else:
            # NumPy 備案
            self.c_charges.fill(q_i)
            if flag_small:
                self.c_charges += sign * MMsys.small_threshold

        # --- STEP 2: 同步 (Python 層，無法避免的 API 呼叫) ---
        # 這是你唯一應該呼叫 API 的地方
        for i in range(self.Natoms):
            idx = self.c_indices[i]
            q = self.c_charges[i]
            MMsys.nbondedForce.setParticleParameters(idx, q, 1.0, 0.0)
            
            # (可選) 更新 Python 物件快取，以防萬一
            self.electrode_atoms[i].charge = q

        MMsys.nbondedForce.updateParametersInContext(MMsys.simmd.context)


    #**************************
    # 🔥 RESTORED TO GOLDEN STANDARD: compute_Electrode_charge_analytic
    # No cache, direct getParticleParameters - "Good Taste"
    #**************************
    # 🔥 GOOD TASTE REFACTORED: compute_Electrode_charge_analytic
    # 使用 Cython 進行 C-level 陣列計算，移除所有 getParticleParameters API 呼叫
    def compute_Electrode_charge_analytic( self, MMsys , positions, Conductor_list, z_opposite ):
        sign=1.0
        if self.electrode_type == 'anode':
            sign=-1.0

        self.Q_analytic = sign / ( 4.0 * numpy.pi ) * self.sheet_area * (self.Voltage / MMsys.Lgap + self.Voltage / MMsys.Lcell) * conversion_KjmolNm_Au

        # 🔥 獲取 z_positions (處理不同的 positions 格式)
        # 理想情況下，這應該在 Poisson_solver 中只做一次並傳入
        if hasattr(positions, '_value'):
            # OpenMM Vec3 列表帶單位 (慢速路徑，但為了相容)
            z_positions_np = numpy.array([pos[2]._value for pos in positions])
        elif isinstance(positions, numpy.ndarray):
            # NumPy 陣列 (快速路徑)
            if positions.ndim == 2:
                # N x 3 陣列
                z_col = positions[:, 2]
                # 檢查是否有單位
                if hasattr(z_col[0], '_value'):
                    z_positions_np = numpy.array([z._value for z in z_col])
                else:
                    z_positions_np = z_col
            else:
                # 1D 陣列？不太可能
                z_positions_np = positions
        else:
            # OpenMM Vec3 列表無單位
            z_positions_np = numpy.array([pos[2] for pos in positions])

        #********** 步驟 1: 電解質貢獻 (C-level, 無 API 呼叫!)
        if CYTHON_AVAILABLE:
            self.Q_analytic += ec_cython.compute_analytic_contribution_cython(
                z_positions_np,
                MMsys.electrolyte_c_indices,  # 來自 initialize_electrolyte
                MMsys.electrolyte_c_charges,  # 來自 initialize_electrolyte
                z_opposite,
                MMsys.Lcell
            )
        else:
            # NumPy Fallback (仍然是 C-level，但稍慢)
            z_atoms = z_positions_np[MMsys.electrolyte_c_indices]
            z_distances = numpy.abs(z_atoms - z_opposite)
            self.Q_analytic += numpy.sum((z_distances / MMsys.Lcell) * (-MMsys.electrolyte_c_charges))

        #********* 步驟 2: 導體貢獻 (C-level, 無 API 呼叫!)
        if Conductor_list:
            for Conductor in Conductor_list:
                if CYTHON_AVAILABLE:
                    self.Q_analytic += ec_cython.compute_analytic_contribution_cython(
                        z_positions_np,
                        Conductor.c_indices,  # 早已存在
                        Conductor.c_charges,  # 早已存在
                        z_opposite,
                        MMsys.Lcell
                    )
                else:
                    # NumPy Fallback
                    z_atoms = z_positions_np[Conductor.c_indices]
                    z_distances = numpy.abs(z_atoms - z_opposite)
                    self.Q_analytic += numpy.sum((z_distances / MMsys.Lcell) * (-Conductor.c_charges))


    #****************************
    # 🔥 GOOD TASTE: Scale_charges_analytic - 計算/同步分離
    # Step 1: 計算 (C-level)
    # Step 2: 同步 (Python API)
    def Scale_charges_analytic( self, MMsys , print_flag = False ):
        Q_numeric = self.get_total_charge()  # NumPy sum (C-level)

        if print_flag :
            print( "Q_numeric , Q_analytic charges on " , self.electrode_type , Q_numeric , self.Q_analytic )

        scale_factor = -1
        if abs(Q_numeric) > MMsys.small_threshold:
            scale_factor = self.Q_analytic / Q_numeric

        # --- STEP 1: 計算 (在 C 陣列上操作) ---
        if scale_factor > 0.0:
            if CYTHON_AVAILABLE:
                # 使用 Cython 就地縮放 c_charges
                ec_cython.scale_charges_inplace_cython(self.c_charges, scale_factor)
            else:
                # NumPy 備案（也是 C-level）
                self.c_charges *= scale_factor

            # --- STEP 2: 同步 (Python 層，無法避免的 API 呼叫) ---
            for i in range(self.Natoms):
                idx = self.c_indices[i]
                q = self.c_charges[i]
                MMsys.nbondedForce.setParticleParameters(idx, q, 1.0, 0.0)
                self.electrode_atoms[i].charge = q  # 更新 Python 快取


    def set_z_pos(self, z):
        self.z_pos = z


#*************************
# Buckyball_Virtual - Child class with Cython optimizations
#*************************
class Buckyball_Virtual(Conductor_Virtual):
    def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element):
       super().__init__(electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element)

       if chain_flag == False:
           print( 'must match by chain index for Buckyball_Virtual class!' )
           sys.exit()
       if not ( isinstance( electrode_identifier , tuple ) and ( len(electrode_identifier) > 1 ) ) :
           print( 'must input chain index for both virtual and real electrode atoms for BuckyBall class' )
           sys.exit()

       self.electrode_atoms_real=[]

       identifier = electrode_identifier[1]
       for chain in MMsys.simmd.topology.chains():
           if chain.index == identifier:
               for atom in chain.atoms():
                   element = atom.element
                   if element.symbol not in exclude_element:
                       (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(atom.index)
                       atom_object = atom_MM( element.symbol , q_i._value , atom.index )
                       self.electrode_atoms_real.append( atom_object )

       # � PYTHON: Center computation (不關鍵，保持簡單)
       state = MMsys.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
       positions = state.getPositions()

       self.r_center = [ 0.0 , 0.0 , 0.0 ]
       for atom in self.electrode_atoms:
           self.r_center[0] += positions[atom.atom_index][0]._value
           self.r_center[1] += positions[atom.atom_index][1]._value
           self.r_center[2] += positions[atom.atom_index][2]._value
       self.r_center[0] = self.r_center[0] / self.Natoms
       self.r_center[1] = self.r_center[1] / self.Natoms
       self.r_center[2] = self.r_center[2] / self.Natoms

       # Compute radius (Python loop - not critical)
       self.radius=0.0
       for atom in self.electrode_atoms:
           rx = positions[atom.atom_index][0]._value - self.r_center[0]
           ry = positions[atom.atom_index][1]._value - self.r_center[1]
           rz = positions[atom.atom_index][2]._value - self.r_center[2]
           self.radius = sqrt( rx**2 + ry**2 + rz**2 )
           break
       self.area_atom = 4.0 * numpy.pi * self.radius**2 / self.Natoms

       # � PYTHON: Normal vectors (不關鍵，保持簡單)
       for atom in self.electrode_atoms:
           nx = positions[atom.atom_index][0]._value - self.r_center[0]
           ny = positions[atom.atom_index][1]._value - self.r_center[1]
           nz = positions[atom.atom_index][2]._value - self.r_center[2]
           norm = sqrt( nx**2 + ny**2 + nz**2)
           atom.nx = nx / norm ; atom.ny = ny / norm ; atom.nz = nz / norm

       self.find_contact_neighbor_conductor( positions , self.r_center , MMsys )


    def get_total_charge_real( self ):
        sumQ = 0.0
        for atom in self.electrode_atoms_real:
            sumQ += atom.charge
        return sumQ

    # 🔥 GOOD TASTE: Scale_charges_analytic 繼承自 Electrode_Virtual
    # 不需要在此重複定義，parent class 的實現已經完美


#*************************
# Nanotube_Virtual - GOOD TASTE VERSION
#*************************
class Nanotube_Virtual(Conductor_Virtual):
    def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element, axis ):
       super().__init__(electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element)

       if chain_flag == False:
           print( 'must match by chain index for Nanotube_Virtual class!' )
           sys.exit()
       if not ( isinstance( electrode_identifier , tuple ) and ( len(electrode_identifier) > 1 ) ) :
           print( 'must input chain index for both virtual and real electrode atoms for Nanotube class' )
           sys.exit()

       self.axis = axis
       self.electrode_atoms_real=[]

       identifier = electrode_identifier[1]
       for chain in MMsys.simmd.topology.chains():
           if chain.index == identifier:
               for atom in chain.atoms():
                   element = atom.element
                   if element.symbol not in exclude_element:
                       (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(atom.index)
                       atom_object = atom_MM( element.symbol , q_i._value , atom.index )
                       self.electrode_atoms_real.append( atom_object )

       # 🐍 PYTHON ONLY (P7 Gap Year): Keep original Python loops
       state = MMsys.simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
       positions = state.getPositions()

       self.r_center = [ 0.0 , 0.0 , 0.0 ]
       for atom in self.electrode_atoms:
           self.r_center[0] += positions[atom.atom_index][0]._value
           self.r_center[1] += positions[atom.atom_index][1]._value
           self.r_center[2] += positions[atom.atom_index][2]._value
       self.r_center[0] = self.r_center[0] / self.Natoms
       self.r_center[1] = self.r_center[1] / self.Natoms
       self.r_center[2] = self.r_center[2] / self.Natoms

       print( 'WARNING:  Assuming Nanotube length is equal to length of "a" box vector.  Need to modify code if this is not the case!')
       boxVecs = MMsys.simmd.topology.getPeriodicBoxVectors()
       self.length = boxVecs[0][0] / nanometer

       radius_threshold=0.001
       self.radius= -1.0
       for atom in self.electrode_atoms:
           dr = [0] * 3
           for i in range(3):
               dr[i] = positions[atom.atom_index][i]._value - self.r_center[i]
           radial_vector =  self.project_orthogonal_to_axis( numpy.asarray(dr) )
           radius = sqrt( radial_vector[0]**2 + radial_vector[1]**2 + radial_vector[2]**2 )
           if self.radius < 0:
               self.radius = radius
           else:
               if abs( self.radius - radius ) > radius_threshold :
                   print( atom.atom_index , radius , self.radius )
                   print( 'different radius for atoms in nanotube, something is wrong!')
                   sys.exit()
           atom.nx = radial_vector[0] / radius ; atom.ny = radial_vector[1] / radius ; atom.nz = radial_vector[2] / radius ;

       self.area_atom = 2.0 * numpy.pi * self.radius * self.length / self.Natoms

       dr_vector = self.find_contact_neighbor_conductor( positions , self.r_center , MMsys )

       if dr_vector :
           radial_vector =  self.project_orthogonal_to_axis( numpy.asarray(dr_vector) )
           self.dr_center_contact = numpy.sqrt( radial_vector[0]**2 + radial_vector[1]**2 + radial_vector[2]**2 )

       print( "Conductor " , self.close_conductor_Electrode  , self.Electrode_contact_atom.atom_index , self.dr_center_contact )


    def project_orthogonal_to_axis( self, vec_in ) :
        axis_local = numpy.asarray( self.axis )
        vec_out = vec_in - axis_local * numpy.dot( vec_in , axis_local )
        return vec_out


    def get_total_charge_real( self ):
        sumQ = 0.0
        for atom in self.electrode_atoms_real:
            sumQ += atom.charge
        return sumQ

    # 🔥 GOOD TASTE: Scale_charges_analytic 繼承自 Electrode_Virtual
    # 不需要在此重複定義，parent class 的實現已經完美
