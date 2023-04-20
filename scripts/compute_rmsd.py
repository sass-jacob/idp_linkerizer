import argparse
from Bio.PDB import PDBParser, PDBIO

parser = argparse.ArgumentParser(description='Remove all non-CA atoms from a PDB file')
parser.add_argument('-input', required=True, help='Input PDB file and path')
parser.add_argument('-output', required=True, help='Output PDB file and path')
args = parser.parse_args()

parser = PDBParser()

structure = parser.get_structure('protein', args.input)

new_structure = []

for model in structure:
    new_model = model.copy()
    for chain in model:
        new_chain = chain.copy()
        for residue in chain:
            new_residue = residue.copy()
            for atom in residue:
                if atom.get_name() == 'CA':
                    new_residue.add(atom.copy())
            new_chain.add(new_residue)
        new_model.add(new_chain)
    new_structure.append(new_model)

io = PDBIO()
io.set_structure(new_structure)
io.save(args.output)
