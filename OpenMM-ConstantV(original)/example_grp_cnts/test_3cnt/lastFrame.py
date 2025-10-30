import mdtraj as md
import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("--output", type=str, help="PDB filename from MD trajectory")
#args = parser.parse_args()

t = md.load("equil_MC.dcd",top="start_drudes.pdb")
#t[-1].save_pdb(str(args.output))
t[-1].save_pdb("lastframe.pdb")
