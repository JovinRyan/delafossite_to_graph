import include
import sys
import os
import pandas as pd
from dgl import save_graphs
from tqdm import tqdm  # cosmetic

import warnings
from pymatgen.io.vasp.inputs import BadPoscarWarning

warnings.filterwarnings("ignore", category=BadPoscarWarning)

if len(sys.argv) > 1:
    directory_path = sys.argv[1]
    csv_filename = "HighThroughput_TM_with_J.csv"
    csv_path = os.path.join(directory_path, csv_filename)

    graph_save_dir = os.path.join(directory_path, "dgl_graphs")

    elemental_dict = include.elemental_CSV_to_nested_dict()

    if os.path.isfile(csv_path):
        print(f"Found CSV file: {csv_path}")
        # You can add your CSV processing code here

        delafossite_data = pd.read_csv(csv_path)
        delafossite_data = delafossite_data.drop_duplicates(delafossite_data.columns[0], keep = 'first')

        print("CSV file contains " + str(len(delafossite_data)) + " entries.")

        features_list = []

        os.makedirs(graph_save_dir, exist_ok=True)

        # Wrap the iterator in tqdm for progress bar display
        for entry in tqdm(delafossite_data.iloc[:, 0], desc="Processing structures"):
            vasp_path = os.path.join(directory_path, entry + ".vasp")
            if os.path.isfile(vasp_path):
                supercell = include.POSCAR_to_supercell(vasp_path)

                g = include.structure_to_dgl_graph(supercell, elemental_dict)

                save_graphs(os.path.join(graph_save_dir, entry + ".bin"), [g])
    else:
        print("CSV file not found.")
