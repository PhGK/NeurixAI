"""Utility functions for feature extraction of Drugs for GNNs."""

# Code taken and adapted from:
# https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/

import pickle
from typing import Dict, List

# Pytorch and Pytorch Geometri
import torch

# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data

# general tools
import numpy as np


def one_hot_encoding(x, permitted_list: List[str]) -> List[int]:
    """
    Map input elements x which are not in the permitted list to the last element of the permitted list.

    Args:
        x: Input element to be encoded.
        permitted_list (List[str]): List of permitted elements.

    Returns
    -------
        List[int]: One-hot encoded representation of the input element.

    Example:
        >>> x = 'A'
        >>> permitted_list = ['A', 'B', 'C']
        >>> encoding = one_hot_encoding(x, permitted_list)
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    return [int(x == s) for s in permitted_list]


def get_atom_features(atom: Chem.Atom, use_chirality: bool = True, hydrogens_implicit: bool = True) -> np.ndarray:
    """
    Take an RDKit atom object as input and gives a 1d-numpy array of atom features as output.

    Args:
        atom (Chem.Atom): RDKit atom object.
        use_chirality (bool, optional): Whether to include chirality information. Default is True.
        hydrogens_implicit (bool, optional): Whether to include implicit hydrogens. Default is True.

    Returns
    -------
        np.ndarray: 1D numpy array of atom features.

    Example:
        >>> atom = Chem.Atom("C")
        >>> features = get_atom_features(atom)
    """
    # define list of permitted atoms
    permitted_list_of_atoms = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Si",
        "P",
        "Cl",
        "Br",
        "Mg",
        "Na",
        "Ca",
        "Fe",
        "As",
        "Al",
        "I",
        "B",
        "V",
        "K",
        "Tl",
        "Yb",
        "Sb",
        "Sn",
        "Ag",
        "Pd",
        "Co",
        "Se",
        "Ti",
        "Zn",
        "Li",
        "Ge",
        "Cu",
        "Au",
        "Ni",
        "Cd",
        "In",
        "Mn",
        "Zr",
        "Cr",
        "Pt",
        "Hg",
        "Pb",
        "Unknown",
    ]

    if not hydrogens_implicit:
        permitted_list_of_atoms = ["H"] + permitted_list_of_atoms

    # compute atom features
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(
        str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
    )
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = (
        atom_type_enc
        + n_heavy_neighbors_enc
        + formal_charge_enc
        + hybridisation_type_enc
        + is_in_a_ring_enc
        + is_aromatic_enc
        + atomic_mass_scaled
        + vdw_radius_scaled
        + covalent_radius_scaled
    )

    if use_chirality:
        chirality_type_enc = one_hot_encoding(
            str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"]
        )
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond: Chem.Bond, use_stereochemistry: bool = True) -> np.ndarray:
    """
    Take an RDKit bond object as input and gives a 1d-numpy array of bond features as output.

    Args:
        bond (Chem.Bond): RDKit bond object.
        use_stereochemistry (bool, optional): Whether to include stereochemistry information. Default is True.

    Returns
    -------
        np.ndarray: 1D numpy array of bond features.

    Example:
        >>> bond = Chem.rdchem.BondType.SINGLE
        >>> features = get_bond_features(bond)
    """
    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def create_pytorch_geometric_graph_data_list_from_smiles(x_smiles: List[str], drug_names: List[str]) -> Dict[str, Data]:
    """
    Convert a list of SMILES strings into a dictionary of PyTorch Geometric Data objects, with drug names as keys.

    Args:
        x_smiles (List[str]): A list of SMILES strings representing molecular structures.
        drug_names (List[str]): A list of drug names associated with the SMILES strings.

    Returns
    -------
        Dict[str, Data]: A dictionary mapping drug names to PyTorch Geometric Data objects representing labeled
                        molecular graphs.

    Example:
        >>> smiles_list = ['CCO', 'CCN']
        >>> drug_names = ['DrugA', 'DrugB']
        >>> data_dict = create_pytorch_geometric_graph_data_list_from_smiles(smiles_list, drug_names)
    """
    data_dict = {}

    for smiles, drug_name in zip(x_smiles, drug_names):

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for k, (i, j) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct Pytorch Geometric data object and add to data dictionary
        data_dict[drug_name] = Data(x=X, edge_index=E, edge_attr=EF)

    return data_dict


def save_dictionary(dictionary: Dict, file_path: str) -> None:
    """
    Save a dictionary to a file using pickle serialization.

    Args:
        dictionary (Dict): The dictionary to be saved.
        file_path (str): The full path including the directory where the file will be saved.

    Example:
        >>> data_dict = {'DrugA': data_object_1, 'DrugB': data_object_2}
        >>> save_dictionary(data_dict, 'data_dict.pkl', '/path/to/save/data_dict.pkl')
    """
    with open(file_path, "wb") as f:
        pickle.dump(dictionary, f)


def load_dictionary(file_path: str) -> Dict:
    """
    Load a dictionary from a file using pickle deserialization.

    Args:
        file_path (str): The name of the file to load the dictionary from.

    Returns
    -------
        Dict: The dictionary loaded from the file.

    Example:
        >>> loaded_data_dict = load_dictionary('data_dict.pkl')
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)
