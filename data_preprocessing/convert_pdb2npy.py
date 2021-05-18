import numpy as np
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
import pickle

def load_protein_structure(fname, center):
    structure = Chem.rdmolfiles.MolFromPDBFile(str(fname))[0]
    conf = structure.GetConformers()[0]
    coords  = np.array(conf.GetPositions())
    atoms = structure.GetAtoms()

    types = []
    for atom in atoms:
        types.append(atom.GetAtomicNum())
    types_array = np.array(types)
    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}

    
    
def load_ligand_structure(fname, center):
    structure = Chem.rdmolfiles.ForwardSDMolSupplier(str(fname))
    conf = structure.GetConformers()[0]
    coords  = np.array(conf.GetPositions())
    atoms = structure.GetAtoms()

    types = []
    for atom in atoms:
        types.append(atom.GetAtomicNum())
    types_array = np.array(types)
    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}

def convert_pdbs(pdb_dir, npy_dir, mols_dir):
    print("Converting PDBs")
    for pdb in tqdm(pdb_dir.glob("[1-9]*")):
        for p in  pdb.glob("protein.pdb"):
            protein = load_protein_structure_np(p, center=False)
            np.save(npy_dir / (p.stem + "_atomxyz.npy"), protein["xyz"])
            np.save(npy_dir / (p.stem + "_atomtypes.npy"), protein["types"])

        for p in  pdb.glob("ligand_sdf"):
            ligand = load_ligand_structure_np(p, center=False)
            np.save(npy_dir / (p.stem + "_atomxyz.npy"), ligand["xyz"])
            np.save(npy_dir / (p.stem + "_atomtypes.npy"), ligand["types"])

def convert_onehot(npy_dir):
    
    ele2num = dict([])
    
    for p in npy_dir.glob("*_atomtypes.npy"):
        types = np.load(p) 
        for t in types:
            if t not in ele2num:
                ele2num[t] = len(ele2num)
                
    with open("element2num.pkl",'wb') as f:
        pickle.dump(ele2num, f)
    
    for p in tqdm(npy_dir.glob("*_atomtypes.npy")):
        types = np.load(p)
        
        types_array = np.zeros((len(types), len(ele2num)))
        for i, t in enumerate(types):
            types_array[i, ele2num[t]] = 1.0

        pdb_id = (p.stem).split('_')[0]
        mol_type = pdb_id = (p.stem).split('_')[1]
        np.save(npy_dir / (pdb_id + '_' + mol_type + "_onehot_types.npy"), types_array)
        
        
npy_dir = Path("../surface_data/raw") / "pdbind_npy"
convert_onehot(npy_dir)