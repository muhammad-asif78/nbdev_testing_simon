import shutil
from pathlib import Path

import nbformat

# Folder containing your notebooks
NOTEBOOKS_DIR = Path("nbs")

# Default module name for nbdev
DEFAULT_EXP = "core"


def tag_notebook(nb_path: Path) -> None:
    """Apply nbdev-style tags to a single notebook in-place.

    - Ensure first cell is `#| default_exp <DEFAULT_EXP>`
    - For each code cell:
        * `#| hide` for magics/shell/installs/demo/print/test cells
        * `#| export` for normal reusable code
      Markdown cells are left unchanged.
    """
    print(f"Processing: {nb_path}")

    # Optional safety: create a backup alongside the notebook
    backup_path = nb_path.with_suffix(nb_path.suffix + ".backup")
    try:
        shutil.copy2(nb_path, backup_path)
    except FileNotFoundError:
        pass

    nb = nbformat.read(nb_path, as_version=4)
    changed = False

    # Ensure top cell is default_exp
    if not nb.cells or not (
        nb.cells[0].cell_type == "code"
        and nb.cells[0].source.strip().startswith(f"#| default_exp {DEFAULT_EXP}")
    ):
        nb.cells.insert(0, nbformat.v4.new_code_cell(f"#| default_exp {DEFAULT_EXP}"))
        changed = True

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        src = cell.source.lstrip()
        if not src:
            continue

        # Respect explicit tags if already present
        if src.startswith("#| export") or src.startswith("#| hide"):
            continue

        lower_src = src.lower()

        # Decide whether to hide this cell
        hide_reasons = (
            src.startswith("!")  # shell/magic
            or src.startswith("%")  # magic
            or "pip install" in lower_src
            or "conda install" in lower_src
            or lower_src.startswith("print(")
            or lower_src.startswith("print ")
            or " demo" in lower_src
            or "test" in lower_src
        )

        if hide_reasons:
            cell.source = "#| hide\n" + cell.source
            changed = True
        else:
            cell.source = "#| export\n" + cell.source
            changed = True

    if changed:
        nbformat.write(nb, nb_path)
        print(f"Tagged notebook: {nb_path}")
    else:
        print(f"No changes needed: {nb_path}")


def main() -> None:
    ipynb_files = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not ipynb_files:
        print(f"No notebooks found in {NOTEBOOKS_DIR}")
        return

    for nb_file in ipynb_files:
        tag_notebook(nb_file)

    print("All notebooks processed.")


if __name__ == "__main__":
    main()
