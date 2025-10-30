import sys
sys.path.append('../../../../Fixed_Voltage_OpenMM/lib/')
from MDAnalysis import *
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from MM_classes_FV import *
#*********** Fixed Voltage routines
from Fixed_Voltage_routines import *
#***************************
import numpy as np
import math
import urllib.request
from add_customnonbond_xml import add_CustomNonbondedForce_SAPTFF_parameters
import matplotlib.pyplot as plt

# for electrode sheets, need to up recursion limit for residue atom matching...
sys.setrecursionlimit(2000)

output_file = "energy_traj" # output file with energies/charges
charge_density_file = "electrolyte_charge_density.xvg" # print this file with charge density
surface_charge_file = "electrode_charge.xvg" # print this file with average charge

u = Universe("start_drudes.pdb", "FV_NVT.dcd")
# this is annoying, but we need two different lists of names, because OpenMM shortens resnames...
electrolyte1 = ( 'BMIM' , 'BF4', 'acnt' )
electrolyte2 = ( 'BMI' , 'BF4', 'acn' )
dz = 0.5 # Angstroms, resolution of charge profile

# area of electrode
area = u.trajectory[0].triclinic_dimensions[0][0] * u.trajectory[0].triclinic_dimensions[1][1]

startFrame = 500
frameCount = u.trajectory.n_frames - startFrame
# this is in case of a long trajectory, for this 
# trajectory we can average every couple timesteps
avgfreq=1
nFramesToAverage = int(frameCount / avgfreq)

# note that u.trajectory[0].dimensions[2] is much
# bigger than actual system because it includes the vaccuum gap
# so this will be 2*3 times more bins than the electrochemical cell

bins = math.ceil( u.trajectory[0].triclinic_dimensions[2][2] / dz )


#************* Create OpenMM objects to get charges.  This uses our MM_system classes...
ffdir='../../../../electrode_ffdir/'
#************************** download SAPT-FF force field files from github
url1 = "https://raw.github.com/jmcdaniel43/SAPT_force_field_OpenMM/master/sapt.xml"
url2 = "https://raw.github.com/jmcdaniel43/SAPT_force_field_OpenMM/master/sapt_residues.xml"
filename1, headers = urllib.request.urlretrieve(url1, "sapt.xml")
filename2, headers = urllib.request.urlretrieve(url2, "sapt_residues.xml")
# add extra CustomNonbonded force parameters to .xml file
add_CustomNonbondedForce_SAPTFF_parameters( xml_base = "sapt.xml" , xml_param =  ffdir + "graph_customnonbonded.xml" , xml_combine = "sapt_add.xml" )

# Initialize: Input list of pdb and xml files, and QMregion_list
MMsys=MM_FixedVoltage( pdb_list = [ 'start_equilibrated.pdb', ] , residue_xml_list = [ 'sapt_residues.xml' , ffdir + 'graph_residue_c.xml', ffdir + 'nanotube9x9_residue_c.xml', ffdir + 'graph_residue_n.xml', ffdir + 'nanotube9x9_residue_n.xml' ] , ff_xml_list = [ 'sapt_add.xml', ffdir + 'graph.xml', ffdir + 'graph_c_freeze.xml', ffdir + 'nanotube9x9_c_freeze.xml' , ffdir + 'graph_n_freeze.xml', ffdir + 'nanotube9x9_n_freeze.xml' ]  )
# if periodic residue, call this
MMsys.set_periodic_residue(True)
#***********  Initialze OpenMM API's, this method creates simulation object
MMsys.set_platform('CPU')   # only 'Reference' platform is currently implemented!

# this creates a list of charges for the electrolyte by loading OpenMM force field files
def create_charge_list( MMsys ):
    # get charges of electrolyte atoms
    charges=[]
    for res in MMsys.simmd.topology.residues():
        if res.name in electrolyte1:
            #print( "resname ", res.name )
            for i in range(len(res._atoms)):
                index = res._atoms[i].index
                (q, sig, eps) = MMsys.nbondedForce.getParticleParameters(index)
                charges.append( q._value )
    return charges

charges = create_charge_list( MMsys )



#*************************** average electrode charge
charges_electrode=[]
with open( output_file ) as f:
    for line in f:
        if "WARNING" in line :
            for i in range(125): # skip 25 interations (5 lines each) of initial charge equilibration
                f.readline()
        if "Anode" in line:
            temp = line.split()
            charges_electrode.append( temp[3] )

# print same chunk of simulation as electrolyte charges
with open( surface_charge_file , 'w' ) as f:
    for i in range( startFrame , len(charges_electrode) ):
        print( charges_electrode[i] , file=f )

#******************************** Average electrolyte charge density

# now select electrolyte atoms
string="resname "
for molecule in electrolyte2:
    string = string + molecule + " "
#print( string )

electrolyte  = u.select_atoms(string)
#print( electrolyte )

charge_density  = [0 for y in range(bins)]

count=0
for i in range(nFramesToAverage):
    currentFrame = startFrame + i * avgfreq
    if currentFrame >= u.trajectory.n_frames:
        break
    count += 1
    u.trajectory[currentFrame]
    for index in range(len(electrolyte.positions)):
        xyz = electrolyte.positions[index]
        q = charges[index]
        i_z = int(xyz[2] / dz )

        # charge density per Ang^3
        charge_density[i_z] += q 

# now normalize
charge_density = np.array(charge_density) / float(count) / area / dz

with open( charge_density_file , 'w' ) as f:
    for i in range(len(charge_density)):
        print( i*dz , charge_density[i] , file=f )


