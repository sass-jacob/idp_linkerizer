import Bio.PDB
import numpy as np

parser = Bio.PDB.PDBParser(QUIET=True)  # quiet avoids printing warnings
structure = parser.get_structure('2beg', '../saved_files/2beg.pdb')  # id of pdb file and location

atoms = [a.parent.parent.id + '-' + str(a.parent.id[1]) + '-' +  a.name for a in structure[0].get_atoms() if a.parent.id[0] == ' ']  # obtained from model '0'
atom_avgs = {}
for atom in atoms:
    atom_avgs[atom] = []
    for model in structure:
        atom_ = atom.split('-')
        coor = model[atom_[0]][int(atom_[1])][atom_[2]].coord
        atom_avgs[atom].append(coor)
    atom_avgs[atom] = sum(atom_avgs[atom]) / len(atom_avgs[atom])  # average

ns = Bio.PDB.StructureBuilder.Structure('id=2beg')  # new structure
ns.add(structure[0])  # add model 0
for atom in ns[0].get_atoms():
    chain = atom.parent.parent
    res = atom.parent
    if res.id[0] != ' ':
        chain.detach_child(res)  # detach hetres
    else:
        coor = atom_avgs[chain.id + '-' + str(res.id[1]) + '-' + atom.name]
        atom.coord = coor

io = Bio.PDB.PDBIO()
io.set_structure(ns)
io.save('../saved_files/avg_2beg.pdb')
