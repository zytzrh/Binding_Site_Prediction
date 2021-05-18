import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.transforms import Compose
import numpy as np
from scipy.spatial.transform import Rotation
import math
import urllib.request
import tarfile
from pathlib import Path
import requests
from data_preprocessing.convert_pdb2npy import convert_pdbs, convert_onehot

tensor = torch.FloatTensor
inttensor = torch.LongTensor

def numpy(x):
    return x.detach().cpu().numpy()

def load_mol_npy(mol_id, npy_dir, center=False):
    # Normalize the point cloud, as specified by the user:
    atom_coords = tensor(np.load(data_dir / (mol_id + "_atomxyz.npy")))
    atom_types = tensor(np.load(data_dir / (mol_id + "_onehot_types.npy")))
 
    if center:
        atom_coords = atom_coords - center_location
    
    molecule_data = Data(
        atom_coords = atom_coords,
        atom_types = atom_types,
    )
    return molecule_data
    
def load_pair(pdb_id, npy_dir):
    mol1_id = pdb_id + '_protein'
    mol2_id = pdb_id + '_ligand'
    
    mol1 = load_mol_npy(mol1_id, npy_dir, center = False)
    mol2 = load_mol_npy(mol2_id, npy_dir, center = False)
    
    mol_dist = ((mol1['xyz'][:,None,:]-mol2['xyz'][None,:,:])**2).sum(-1).sqrt()
    mol_dist = mol_dist < 2.0   # cut off distance , if <2A atoms are labeled as interacting
    y_1 = (mol_dist.sum(1)>0).to(torch.float).reshape(-1,1)
    y_2 = (mol_dist.sum(0)>0).to(torch.float).reshape(-1,1)

    pair_data = PairData(
        xyz_p1 = mol1["xyz"], 
        xyz_p2 = mol2["xyz"], 
        y_p1 = y_1, 
        y_p2 = y_2,
        atom_coords_p1 = mol1["atom_coords"],
        atom_coords_p2 = mol2["atom_coords"],
        atom_center1 = torch.mean(mol1["atom_coords"], axis=0, keepdims=True)
        atom_center2 = torch.mean(mol2["atom_coords"], axis=0, keepdims=True)
        atom_types_p1=mol1["atom_types"],
        atom_types_p2=mol2["atom_types"]
    )
    
    return pair_data
    
class PairData(Data):
    def __init__(
        self,
        y_p1 = None,
        y_p2 = None,
        atom_coords_p1=None,
        atom_coords_p2=None,
        atom_types_p1=None,
        atom_types_p2=None,
        atom_center1=None,
        atom_center2=None,
        rand_rot1=None,
        rand_rot2=None,
    ):
        super().__init__()
        self.y_p1 = y_p1
        self.y_p2 = y_p2
        self.atom_coords_p1 = atom_coords_p1
        self.atom_coords_p2 = atom_coords_p2
        self.atom_types_p1 = atom_types_p1
        self.atom_types_p2 = atom_types_p2
        self.atom_center1 = atom_center1
        self.atom_center2 = atom_center2
        self.rand_rot1 = rand_rot1
        self.rand_rot2 = rand_rot2

class InteractionPair(InMemoryDataset):
    url = ""

    def __init__(self, root, ppi=False, train=True, transform=None, pre_transform=None):
        super(ProteinPairsSurfaces, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        self.ppi = ppi
        
    @property
    def raw_file_names(self):
        return "pdbbind_v2019_refined.tar"

    @property
    def processed_file_names(self):
        if not self.ppi:
            file_names = [
                "training_pairs_data.pt",
                "testing_pairs_data.pt",
                "training_pairs_data_ids.npy",
                "testing_pairs_data_ids.npy",
            ]
        else:
            file_names = [
                "training_pairs_data_ppi.pt",
                "testing_pairs_data_ppi.pt",
                "training_pairs_data_ids_ppi.npy",
                "testing_pairs_data_ids_ppi.npy",
            ]
        return file_names

    def download(self):
        url = ''
        target_path = self.raw_paths[0]
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())
                
        #raise RuntimeError(
        #    "Dataset not found. Please download {} from {} and move it to {}".format(
        #        self.raw_file_names, self.url, self.raw_dir
        #    )
        #)

    def process(self):
        mols_dir = Path(self.root) / "raw" / "pdbind_refined"
        surf_dir = None
        pair_dir = Path(self.root) / "raw" / "pdbind_npy"
        lists_dir = Path('./lists')

        # Untar surface files
        if not (mols_dir.exists()):
            tar = tarfile.open(self.raw_paths[0])
            tar.extractall(self.raw_dir)
            tar.close()

        # last edit
        
        if not pair_dir.exists():
            pair_dir.mkdir(parents=False, exist_ok=False)
            convert_pdbs(mols_dir, pair_dir, None)
            convert_onehot(pair_dir)

        with open(lists_dir / "train.txt") as f_tr, open(
            lists_dir / "test.txt"
        ) as f_ts:
            training_list = sorted(f_tr.read().splitlines())
            testing_list = sorted(f_ts.read().splitlines())
           
        # Read data into huge `Data` list.
        training_pairs_data = []
        training_pairs_data_ids = []
        training_pairs_affinity = []
        
        for p in training_list :
            try: 
                protein_ligand_pair = load_pair(p, pair_dir)
                affinity = load_affinity(p)
            except FileNotFoundError
                continue
            training_pairs_data.append(protein_ligand_pair)
            training_pairs_data_ids.append(p)
            training_pairs_affinity.append(affinity)
        
        testing_pairs_data = []
        testing_pairs_data_ids = []
        testing_pairs_affinity = []
        
        for p in testing_list :
            try: 
                protein_ligand_pair = load_pair(p, pair_dir)
                affinity = load_affinity(p)
            except FileNotFoundError
                continue
            testing_pairs_data.append(protein_ligand_pair)
            testing_pairs_data_ids.append(p)
            testing_pairs_affinity.append(affinity)

        if self.pre_filter is not None:
            training_pairs_data = [
                data for data in training_pairs_data if self.pre_filter(data)
            ]
            testing_pairs_data = [
                data for data in testing_pairs_data if self.pre_filter(data)
            ]

        if self.pre_transform is not None:
            training_pairs_data = [
                self.pre_transform(data) for data in training_pairs_data
            ]
            testing_pairs_data = [
                self.pre_transform(data) for data in testing_pairs_data
            ]

        training_pairs_data, training_pairs_slices = self.collate(training_pairs_data)
        torch.save(
            (training_pairs_data, training_pairs_slices), self.processed_paths[0]
        )
        np.save(self.processed_paths[2], training_pairs_data_ids)
        testing_pairs_data, testing_pairs_slices = self.collate(testing_pairs_data)
        torch.save((testing_pairs_data, testing_pairs_slices), self.processed_paths[1])
        np.save(self.processed_paths[3], testing_pairs_data_ids)
