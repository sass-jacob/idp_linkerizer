from Bio import pairwise2, PDB
from Bio.PDB import Superimposer

def get_coordinates(pdb_file, chain_ids):
    """
    Extracts alpha carbon coordinates for residues 17-42 for each chain in pdb_file.
    
    Args:
    pdb_file (str): path to PDB file
    chain_ids (list): list of chain IDs to extract coordinates for
    
    Returns:
    dict: dictionary containing chain IDs as keys and alpha carbon coordinates as values
    """
    coords = {}
    parser = PDBParser()
    for chain_id in chain_ids:
        coords[chain_id] = []
    with open(pdb_file, 'r') as f:
        structure = parser.get_structure('pdb', f)
        for model in structure:
            for chain in model:
                if chain.id in chain_ids:
                    for residue in chain:
                        if residue.get_id()[1] >= 17 and residue.get_id()[1] <= 42:
                            try:
                                coords[chain.id].append(residue['CA'].get_coord())
                            except KeyError:
                                pass
    return coords

# Load reference and other pdb structures
ref_pdb = PDB.PDBParser().get_structure("ref", "../saved_files/avg_2beg.pdb")
other_pdb = PDB.PDBParser().get_structure("other", "other.pdb")

# Get alpha carbon atoms from both structures
ref_atoms = [a for a in ref_pdb.get_atoms() if a.get_id() == 'CA']
other_atoms = [a for a in other_pdb.get_atoms() if a.get_id() == 'CA']

# Get the residues common to both structures
ref_residues = {res.id for res in ref_pdb.get_residues()}
common_residues = [res for res in other_pdb.get_residues() if res.id in ref_residues]

# Get alpha carbons for only the common residues
ref_atoms = [a for a in ref_atoms if a.get_parent().id in ref_residues]
other_atoms = [a for a in other_atoms if a.get_parent() in common_residues]

# Perform superimposition and calculate RMSD
sup = Superimposer()
sup.set_atoms(ref_atoms, other_atoms)
rmsd = sup.rms

print("RMSD:", rmsd)
