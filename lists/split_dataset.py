import numpy as np
from pathlib import Path

def split_dataset(data_dir, list_dir):
    
    pdbs = set([])
    
    for p in data_dir.glob("*"):
        mol = p.stem
        pdb_id = mol.split('_')[0]
        pdbs.add(pdb_id)
    
    pdbs = sorted(list(pdbs))
    
    num_val = len(pdbs)//12
    num_test = len(pdbs)//12
    num_train = len(pdbs) - num_val - num_test

    pdbs_train = pdbs[:num_train]
    pdbs_val = pdbs[num_train: num_train + num_val]
    pdbs_test = pdbs[num_train + num_val: ]

    train_f = open(list_dir / "train.txt", 'w')
    val_f = open(list_dir / "val.txt", 'w')
    test_f = open(list_dir / "test.txt", 'w')
    
    train_f.write(('\n').join(pdbs_train))
    val_f.write(('\n').join(pdbs_val))
    test_f.write(('\n').join(pdbs_test))

# parse dataset
mol_dir = Path("../surface_data") / "raw" / "pdbind_mol"
list_dir = Path(".")
split_dataset(mol_dir , list_dir)