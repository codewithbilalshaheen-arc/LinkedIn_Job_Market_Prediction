"""
Standalone Jupyter Notebook Executor and Output Populator.
Executes notebook code cells, captures stdout/stderr and matplotlib plots,
and writes full execution outputs and execution counts into the .ipynb JSON.
Does not require nbconvert or jupyter packages.
"""

import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOTEBOOKS_DIR = Path("Notebooks")


def execute_and_populate(notebook_path: Path):
    print(f"\n==================================================")
    print(f"Executing and populating: {notebook_path.name}")
    print(f"==================================================")

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Global and local execution namespace
    exec_globals = {
        "__name__": "__main__",
        "__file__": str(notebook_path.resolve())
    }

    # Change working directory to Notebooks directory so relative paths work
    orig_cwd = os.getcwd()
    os.chdir(NOTEBOOKS_DIR)
    if ".." not in sys.path:
        sys.path.insert(0, "..")

    execution_counter = 1

    try:
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue

            source_code = "".join(cell["source"])
            cell["execution_count"] = execution_counter
            cell["outputs"] = []

            # Capture stdout and stderr
            stdout_io = io.StringIO()
            stderr_io = io.StringIO()

            plt.close("all")

            try:
                with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(stderr_io):
                    exec(source_code, exec_globals)
                success = True
            except Exception as e:
                success = False
                err_msg = f"{type(e).__name__}: {e}"
                print(f"  [Cell {execution_counter}] Exception: {err_msg}")
                cell["outputs"].append({
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [err_msg]
                })

            stdout_text = stdout_io.getvalue()
            if stdout_text:
                cell["outputs"].append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [line + "\n" for line in stdout_text.rstrip("\n").split("\n")]
                })

            # Check if any matplotlib figures were generated
            fig_nums = plt.get_fignums()
            for fig_num in fig_nums:
                fig = plt.figure(fig_num)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight")
                img_buf.seek(0)
                img_b64 = base64.b64encode(img_buf.read()).decode("utf-8")
                cell["outputs"].append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {}
                })
            plt.close("all")

            status_str = "OK" if success else "ERR"
            print(f"  [Cell {execution_counter}] {status_str} | stdout lines: {len(stdout_text.splitlines())}")
            execution_counter += 1

    finally:
        os.chdir(orig_cwd)

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"--> Saved executed outputs for {notebook_path.name}")


if __name__ == "__main__":
    notebooks = [
        NOTEBOOKS_DIR / "01_Exploratory_Data_Analysis.ipynb",
        NOTEBOOKS_DIR / "02_Data_Preprocessing.ipynb",
        NOTEBOOKS_DIR / "03_CNN_Model_Implementation.ipynb",
        NOTEBOOKS_DIR / "04_Model_Training_and_Evaluation.ipynb",
        NOTEBOOKS_DIR / "05_Proposed_Research_Extension.ipynb"
    ]
    for nb in notebooks:
        if nb.exists():
            execute_and_populate(nb)
