from typing import Sequence
import argparse
from pathlib import Path
import toml
import dolfin


def run(mesh_path: Path, output_path: Path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(str(mesh_path)) as f:
        f.read(mesh)

    msg = f"Mesh has {mesh.num_vertices()} vertices."
    print(msg)
    (output_path / "output.txt").write_text(msg)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simcardems CLI")
    parser.add_argument("config-file", type=Path, help="Config file")

    args = vars(parser.parse_args(argv))
    config_file = toml.loads(args["config-file"].read_text())
    run(**config_file)
    return 0
