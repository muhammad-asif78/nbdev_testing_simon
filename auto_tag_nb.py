import shutil
import nbformat

notebook_path = "nbs/00_core.ipynb"
backup_path = "nbs/00_core.backup.ipynb"

# Optional safety: create a backup of the original notebook
try:
    shutil.copy2(notebook_path, backup_path)
except FileNotFoundError:
    pass

# Load notebook
nb = nbformat.read(notebook_path, as_version=4)

# Ensure top cell has default_exp core
if nb.cells:
    top_cell = nb.cells[0]
    if not (top_cell.cell_type == "code" and top_cell.source.strip().startswith("#| default_exp core")):
        nb.cells.insert(0, nbformat.v4.new_code_cell("#| default_exp core"))
else:
    nb.cells = [nbformat.v4.new_code_cell("#| default_exp core")]

for cell in nb.cells:
    # Only modify code cells
    if cell.cell_type != "code":
        continue

    source = cell.source.lstrip()
    if not source:
        continue

    # If already explicitly tagged, skip (assume user knows better)
    if source.startswith("#| export") or source.startswith("#| hide"):
        continue

    lower_src = source.lower()

    # Criteria for HIDE:
    # - Jupyter magics or shell commands (! or % at start)
    # - pip/conda/install commands
    # - obvious demo / print / test code
    hide_reasons = (
        source.startswith("!") or
        source.startswith("%") or
        "pip install" in lower_src or
        "conda install" in lower_src or
        lower_src.startswith("print(") or
        lower_src.startswith("print ") or
        " demo" in lower_src or
        "test" in lower_src
    )

    if hide_reasons:
        cell.source = "#| hide\n" + cell.source
    else:
        cell.source = "#| export\n" + cell.source

# Save notebook
nbformat.write(nb, notebook_path)
print(f"Notebook {notebook_path} tagged successfully!")
