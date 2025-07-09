import include
import sys
import os
import pandas as pd
from dgl import save_graphs
from tqdm import tqdm
import warnings
from pymatgen.io.vasp.inputs import BadPoscarWarning
import concurrent.futures

warnings.filterwarnings("ignore", category=BadPoscarWarning)

def process_entry(entry_data):
    """
    Processes one entry: reads VASP file, converts to DGL graph, saves to disk.
    entry_data is a tuple: (entry_name, directory_path, graph_save_dir, elemental_dict)
    """
    entry, directory_path, graph_save_dir, elemental_dict = entry_data
    vasp_path = os.path.join(directory_path, entry + ".vasp")
    save_path = os.path.join(graph_save_dir, entry + ".bin")

    try:
        if not os.path.isfile(vasp_path):
            return f"Skipped (missing): {entry}"

        supercell = include.POSCAR_to_supercell(vasp_path)
        g = include.structure_to_dgl_graph(supercell, elemental_dict)
        save_graphs(save_path, [g])
        return f"Success: {entry}"

    except Exception as e:
        return f"Failed: {entry} ({str(e)})"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
        csv_filename = "HighThroughput_TM_with_J.csv"
        csv_path = os.path.join(directory_path, csv_filename)
        graph_save_dir = os.path.join(directory_path, "dgl_graphs")

        if not os.path.isfile(csv_path):
            print("CSV file not found.")
            sys.exit(1)

        print(f"Found CSV file: {csv_path}")
        delafossite_data = pd.read_csv(csv_path)
        delafossite_data = delafossite_data.drop_duplicates(delafossite_data.columns[0], keep='first')
        print("CSV file contains", len(delafossite_data), "entries.")

        os.makedirs(graph_save_dir, exist_ok=True)
        elemental_dict = include.elemental_CSV_to_nested_dict()

        # Prepare list of arguments for each entry
        entry_args = [
            (entry, directory_path, graph_save_dir, elemental_dict)
            for entry in delafossite_data.iloc[:, 0]
        ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            results = list(tqdm(
                executor.map(process_entry, entry_args),
                total=len(entry_args),
                desc="Processing structures"
            ))

        # Print or log the results (optional)
        for r in results:
            print(r)

    else:
        print("Usage: python main.py <directory_path>")
