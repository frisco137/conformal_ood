import json
import sys

def parse_notebook(filepath):
    with open(filepath, "r") as f:
        nb = json.load(f)
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            print(f"=== Cell {i} ===")
            print("".join(cell.get("source", [])))
            print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parse_notebook(sys.argv[1])
    else:
        print("Please provide the path to a notebook file.")
