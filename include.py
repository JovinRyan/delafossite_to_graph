import numpy as np
import pandas as pd
import re
import os
import dgl
import torch
from collections import defaultdict
from pymatgen.io.vasp import Poscar
from scipy.signal import find_peaks
from pymatgen.core import Structure
from matminer.featurizers.structure import RadialDistributionFunction
from sklearn.preprocessing import LabelEncoder

os.environ['DGLBACKEND'] = 'pytorch'

def POSCAR_to_supercell(input_POSCAR_file: str, supercell_matrix = [1, 1, 1]):
  structure = Poscar.from_file(input_POSCAR_file).structure

  return structure.make_supercell(supercell_matrix)


def compute_per_atom_rdfs(structure: Structure, cutoff=10.0, bin_size=0.05):
    """
    Computes per-atom RDFs for a given pymatgen structure.

    Args:
        structure (Structure): Pymatgen structure object (can be supercell).
        cutoff (float): Maximum distance to consider (Å).
        bin_size (float): Size of each RDF bin (Å).

    Returns:
        r_values (np.ndarray): Bin centers for RDF.
        rdfs (np.ndarray): Shape [n_atoms, n_bins] — per-atom RDFs.
    """
    n_atoms = len(structure)
    n_bins = int(cutoff / bin_size)
    r_values = np.linspace(0, cutoff, n_bins)
    rdfs = np.zeros((n_atoms, n_bins))

    # Precompute all neighbors with periodic images
    all_neighbors = structure.get_all_neighbors(r=cutoff, include_index=True)

    for i in range(n_atoms):
        neighbors = all_neighbors[i]
        distances = [dist for site, dist, j, image in neighbors]
        hist, _ = np.histogram(distances, bins=n_bins, range=(0, cutoff))
        rdfs[i] = hist

    # Normalize RDFs (optional)
    shell_volumes = 4/3 * np.pi * (
        np.power(np.linspace(bin_size, cutoff, n_bins), 3) -
        np.power(np.linspace(0, cutoff - bin_size, n_bins), 3)
    )
    avg_density = len(structure) / structure.volume
    normalization = avg_density * shell_volumes
    rdfs = rdfs / normalization[np.newaxis, :]

    return r_values, rdfs

def get_element_cutoff_dict_from_rdf(
    structure: Structure,
    cutoff=10.0,
    bin_size=0.05,
    padding=0.1  # Å padding to second peak
):
    """
    Computes cutoff distances for each element based on the second RDF peak + padding.

    Parameters:
    -----------
    structure : pymatgen Structure
        A pre-expanded supercell structure.
    cutoff : float
        Maximum RDF calculation radius in Å.
    bin_size : float
        RDF histogram bin size in Å.
    padding : float
        Extra distance added to the second RDF peak.

    Returns:
    --------
    cutoff_dict : dict
        Maps element symbols to cutoff distances in Å.
    """

    # Compute per-atom RDFs
    rdf_calc = RadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
    r_vals, per_atom_rdfs = compute_per_atom_rdfs(structure, cutoff=cutoff, bin_size=bin_size)

    # Group RDFs by element
    elements = [site.specie.symbol for site in structure]
    rdf_groups = defaultdict(list)
    for i, elem in enumerate(elements):
        rdf_groups[elem].append(per_atom_rdfs[i])

    # Compute mean RDFs
    mean_rdfs = {elem: np.mean(rdfs, axis=0) for elem, rdfs in rdf_groups.items()}

    # Find 2nd RDF peak + padding
    cutoff_dict = {}
    for elem, rdf_curve in mean_rdfs.items():
        peaks, _ = find_peaks(rdf_curve)
        if len(peaks) >= 2:
            second_peak = r_vals[peaks[1]]
        elif len(peaks) == 1:
            second_peak = r_vals[peaks[0]]
        else:
            second_peak = cutoff  # fallback

        cutoff_dict[elem] = second_peak + padding

    return cutoff_dict

def encode_ElectronConfiguration(electronic_configuration : str):
  """
  Converts an electronic configuration string into valence electron counts.

  Args:
    electronic_configuration (str): Electronic configuration of an element,
    e.g., "1s2 2s2 2p6 3s2 3p6 4s1".

  Returns:
    tuple: A 5-element tuple containing:
      - valence_electrons (int): Total number of valence electrons.
      - s_electrons (int): Number of valence electrons in the s orbital.
      - p_electrons (int): Number of valence electrons in the p orbital.
      - d_electrons (int): Number of valence electrons in the d orbital.
      - f_electrons (int): Number of valence electrons in the f orbital.
  """

  # Remove any annotations and noble gas core notations like [He], [Rn]
  clean_config = re.sub(r'\[.*?\]', '', electronic_configuration)
  clean_config = re.sub(r'\(.*?\)', '', clean_config)

  electronic_configuration_list = clean_config.strip().split()

  s_electrons = 0
  p_electrons = 0
  d_electrons = 0
  f_electrons = 0

  for block in electronic_configuration_list:
      match = re.match(r'(\d+)([spdf])(\d+)', block)
      if match:
          shell, orbital, count = match.groups()
          count = int(count)
          if orbital == 's':
              s_electrons = count
          elif orbital == 'p':
              p_electrons = count
          elif orbital == 'd':
              d_electrons = count
          elif orbital == 'f':
              f_electrons = count

  valence_electrons = s_electrons + p_electrons + d_electrons + f_electrons

  return (valence_electrons, s_electrons, p_electrons, d_electrons, f_electrons)

def elemental_CSV_to_nested_dict(csv_file_name : str = "PubChemElements_all.csv"):
  '''
  Reads elemental Data from National Center for Biotechnology Information. Periodic Table of Elements. https://pubchem.ncbi.nlm.nih.gov/periodic-table/. Accessed May 30, 2025.
  Used for informing feature vector for Machine Learning.
  CSV file contains columns: ["AtomicNumber", "Symbol", "Name", "AtomicMass", "CPKHexColor", "ElectronConfiguration", "Electronegativity", "AtomicRadius",
  "IonizationEnergy", "ElectronAffinity", "OxidationStates", "States", "MeltingPoint", "BoilingPoint", "Density", "GroupBlock", "YearDiscovered"]

  Args:
    elemental_csv (str): Elemental Data CSV file name.

  Returns:
    elemental_df (pd.Dataframe): Dataframe with columns: "["AtomicNumber", "Symbol", "AtomicMass", "element_valence_e", "element_s_e", "element_p_e", "element_d_e", "element_f_e", "Electronegativity", "AtomicRadius", "IonizationEnergy", "ElectronAffinity", "OxidationStates"]"
  '''

  relevant_columns = ["AtomicNumber", "Symbol", "AtomicMass", "element_valence_e", "element_s_e", "element_p_e", "element_d_e", "element_f_e", "Electronegativity", "AtomicRadius", "IonizationEnergy", "ElectronAffinity", "OxidationStates"]
  elemental_CSV = pd.read_csv(csv_file_name)

  element_valence_e_list = []
  element_s_e_list = []
  element_p_e_list = []
  element_d_e_list = []
  element_f_e_list = []

  for i in range(len(elemental_CSV)):
    val_e, s_e, p_e, d_e, f_e =  encode_ElectronConfiguration(elemental_CSV["ElectronConfiguration"][i])
    element_valence_e_list.append(val_e)
    element_s_e_list.append(s_e)
    element_p_e_list.append(p_e)
    element_d_e_list.append(d_e)
    element_f_e_list.append(f_e)

  elemental_CSV = elemental_CSV.assign(
    element_valence_e=element_valence_e_list,
    element_s_e=element_s_e_list,
    element_p_e=element_p_e_list,
    element_d_e=element_d_e_list,
    element_f_e=element_f_e_list)

  elemental_CSV = elemental_CSV[relevant_columns].fillna(0)

  elemental_CSV["AtomicRadius"] = elemental_CSV["AtomicRadius"] * 0.01 # Converting from pm to Angstrom

  return elemental_CSV

def structure_to_dgl_graph(structure: Structure,
                           elemental_df: pd.DataFrame,
                           cutoff=None,
                           cutoff_padding=0.1,
                           rdf_cutoff=10.0,
                           rdf_bin_size=0.05):
    """
    Converts a Pymatgen supercell structure into a DGLGraph with rich node and edge features.

    Parameters:
    -----------
    structure : pymatgen Structure
        Supercell structure to convert.
    elemental_df : pd.DataFrame
        Precomputed elemental feature dataframe (from `elemental_CSV_to_nested_dict()`).
    cutoff : dict or None
        Optional manual dictionary of cutoff distances per element symbol.
    cutoff_padding : float
        Å padding to be added to RDF-based cutoff if computed.
    rdf_cutoff : float
        RDF computation cutoff for automatic cutoff detection.
    rdf_bin_size : float
        RDF bin size in Å.

    Returns:
    --------
    g : dgl.DGLGraph
        DGL graph object with node and edge features.
    """

    elements = [site.specie.symbol for site in structure]
    n_atoms = len(structure)

    # 1. Compute cutoff if not given
    if cutoff is None:
        cutoff = get_element_cutoff_dict_from_rdf(structure,
                                                  cutoff=rdf_cutoff,
                                                  bin_size=rdf_bin_size,
                                                  padding=cutoff_padding)

    # 2. Edge building
    edge_src, edge_dst, edge_length, edge_image = [], [], [], []
    all_neighbors = structure.get_all_neighbors(r=max(cutoff.values()), include_index=True)

    for i in range(n_atoms):
        elem_i = elements[i]
        elem_cutoff = cutoff.get(elem_i, max(cutoff.values()))

        for site, dist, j, image in all_neighbors[i]:
            if dist <= elem_cutoff:
                edge_src.append(i)
                edge_dst.append(j)
                edge_length.append(dist)
                edge_image.append(image)

    # 3. Node features
    feature_columns = [
        "AtomicNumber", "AtomicMass", "element_valence_e", "element_s_e", "element_p_e",
        "element_d_e", "element_f_e", "Electronegativity", "AtomicRadius",
        "IonizationEnergy", "ElectronAffinity"
    ]

    atomic_features = []
    for el in elements:
        row = elemental_df[elemental_df["Symbol"] == el]
        if row.empty:
            raise ValueError(f"Element {el} not found in elemental dataframe.")
        atomic_features.append(row[feature_columns].values[0])

    node_features = torch.tensor(np.array(atomic_features), dtype=torch.float32)

    # 4. Graph construction
    src = torch.tensor(edge_src, dtype=torch.int64)
    dst = torch.tensor(edge_dst, dtype=torch.int64)
    g = dgl.graph((src, dst), num_nodes=n_atoms)

    g.ndata['feat'] = node_features

    g.edata['length'] = torch.tensor(edge_length, dtype=torch.float32).unsqueeze(1)
    g.edata['periodicity'] = torch.tensor(edge_image, dtype=torch.int8)  # shape: [num_edges, 3]

    g = dgl.add_self_loop(g)
    g = dgl.to_bidirected(g, copy_ndata=True)

    label_encoder = LabelEncoder()

    g.ndata["element_id"] = torch.tensor(label_encoder.fit_transform(elements))

    return g
