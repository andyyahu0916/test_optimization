from sys import stdout
from MDAnalysis import *
import numpy
import MDAnalysis.analysis.distances as distances
from numpy import linalg as LA
import matplotlib.pyplot as plt
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("--volt", type=str, help="PDB file with initial positions")
#args = parser.parse_args()
ion_ex='Tf2'
atoms_ex=('Ctf','Ctf1')
exclusions=(len(atoms_ex),len(atoms_ex))
# create string to search for atoms, note all atoms in same molecule must be contiguous in list for exclusions to work
string=""
for i in range(len(atoms_ex)):
    ele=atoms_ex[i]
    string = string + "name %s and resname %s" % (ele, ion_ex)
    if i < len(atoms_ex)-1:
        string = string + " or "


traj='md_nvt300.dcd'
top='start_drudes300.pdb'
u = Universe(top, traj)

framestart = 6000
frameCount = 4000
frameend = framestart + frameCount

grp1 = u.select_atoms("resname grp and segid A and name C1")
grp2 = u.select_atoms("resname grp and segid E and name C1")
Lcell = float(grp2.positions[:,2] - grp1.positions[:,2])
ion1 =  u.select_atoms( string )
#ion1 = u.select_atoms("resname Tf2 and (name Ctf*) ")
#ion2 = u.select_atoms("name N1* and resname BMI")
ion2 = u.select_atoms("name C1 or name C2* and resname BMI")

dmin, dmax = 0.0, 15.0
bin_width = 0.2
n_bins = int(dmax/bin_width)
layer1_cutoff = 6.

# Tf2-BMI pairs
rdf_ion1_ion2, edges = numpy.histogram([0], bins=n_bins, range=(dmin, dmax))
rdf_ion1_ion2 = numpy.array(rdf_ion1_ion2,dtype=numpy.float64)
rdf_ion1_ion2 *= 0
n_ion_pairs_12 = 0.
# Tf2-Tf2 pairs
rdf_ion1_ion1, edges = numpy.histogram([0], bins=n_bins, range=(dmin, dmax))
rdf_ion1_ion1 = numpy.array(rdf_ion1_ion1,dtype=numpy.float64)
rdf_ion1_ion1 *= 0
n_ion_pairs_11 = 0.
for ts in u.trajectory[framestart:frameend]:
    box = ts.dimensions
    group1_crd = grp1.positions
    ion1_crd = ion1.positions
    ion2_crd = ion2.positions
    ion1_resids = ion1.resids

    ion1_layer1_pos = []
    ion1_layer1_resid = []
    for ion1_i in ion1:
        #print(dir(ion1_i))
        ion1_i_crd = ion1_i.position
        ion1_i_residx = ion1_i.resid
        if (abs(float(ion1_i_crd[2]) - float(group1_crd[0,2])) < layer1_cutoff):
            ion1_layer1_pos.append(ion1_i.position)
            ion1_layer1_resid.append(ion1_i.resid)
    ion1_layer1_pos = numpy.array(ion1_layer1_pos,dtype=numpy.float64)
    ion1_layer1_resid = numpy.array(ion1_layer1_resid,dtype=numpy.float64)
    #print(ion1_layer1_resid, len(ion1_layer1_resid))
    # Tf2-BMI pairs
    pairs_12, dist_12 = distances.capped_distance(  ion1_layer1_pos, ion2_crd, dmax , min_cutoff=dmin, box=ts.dimensions )
    new_rdf_12, edges = numpy.histogram(numpy.ravel(dist_12), bins=n_bins, range=(dmin, dmax))
    n_ion_pairs_12 += len(dist_12)
    new_rdf_12 = numpy.array(new_rdf_12,dtype=numpy.float64)
    rdf_ion1_ion2 += new_rdf_12

    # Tf2-Tf2 pairs

    pairs_11, dist_11 = distances.capped_distance(  ion1_layer1_pos, ion1_crd, dmax , min_cutoff=dmin, box=ts.dimensions )
    if exclusions is not None:
        idxA = [ int(ion1_layer1_resid[pairs_11[i,0]]) for i in range(len(pairs_11))]
        idxA = numpy.array(idxA, dtype=numpy.float64)
        idxB = [int(ion1_resids[pairs_11[i,1]]) for i in range(len(pairs_11)) ]
        idxB = numpy.array(idxB, dtype=numpy.float64)
        #print("idxA", idxA)
        #print("idxB", idxB)
        mask = numpy.where(idxA != idxB)[0]
        nomask = numpy.where(idxA == idxB)[0]
        #print("mask", mask, len(mask))
        #print("nomask", nomask, len(nomask))
        dist_11 = dist_11[mask]
    new_rdf_11, edges = numpy.histogram(numpy.ravel(dist_11), bins=n_bins, range=(dmin, dmax))
    n_ion_pairs_11 += len(dist_11)
    new_rdf_11 = numpy.array(new_rdf_11,dtype=numpy.float64)
    rdf_ion1_ion1 += new_rdf_11

# Normalize RDF
n_ion_pairs_12 = n_ion_pairs_12 / frameCount
n_ion_pairs_11 = n_ion_pairs_11 / frameCount
vol = (4./ 3.) * numpy.pi  * (numpy.power(edges[1:],3)-numpy.power(edges[:-1],3))
vol_sphere =  (4 / 3) * numpy.pi *dmax**3
rdf_ion1_ion2 = rdf_ion1_ion2 / n_ion_pairs_12 / (vol*frameCount)*vol_sphere
rdf_ion1_ion1 = rdf_ion1_ion1 / n_ion_pairs_11 / (vol*frameCount)*vol_sphere

newrdf_12 = numpy.array(rdf_ion1_ion2)
newrdf_11 = numpy.array(rdf_ion1_ion1)
edges = 0.5 * (edges[1:] + edges[:-1])
newedge = numpy.array(edges)
numpy.savetxt('rdf_1stlayerCtf_Cbmi.dat', numpy.column_stack((newedge,newrdf_12)), fmt="%10.6G")
numpy.savetxt('rdf_1stlayerCtf_Ctf.dat', numpy.column_stack((newedge,newrdf_11)), fmt="%10.6G")

#plt.figure()
#plt.plot(newedge,newrdf_12,color="k",linestyle='-')
#plt.plot(newedge,newrdf_11,color="r",linestyle='-')
#plt.xlabel("distance ($\AA$)",fontsize=14)
#plt.ylabel("g(r)",fontsize=14)
#plt.legend(('Ctf--BMI', 'Ctf--Ctf'), loc='best',fontsize=13,frameon=False)
##plt.savefig('rdf_1stlayerTf2_cell'+str(args.volt)+'V.png')
#plt.savefig('rdf_1stlayerCtf_cell.png')
