import mdtraj as md
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, help="PDB filename from MD trajectory")
args = parser.parse_args()

t = md.load("md_nvt300.dcd",top="start_drudes300.pdb")
t[-10].save_pdb(str(args.output))
