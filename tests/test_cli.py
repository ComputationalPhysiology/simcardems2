import dolfin
import toml
from simcardems2.cli import main


def test_cli(tmp_path):
    mesh = dolfin.UnitCubeMesh(5, 5, 5)
    with dolfin.XDMFFile(str(tmp_path / "mesh.xdmf")) as f:
        f.write(mesh)
    config = {"mesh_path": str(tmp_path / "mesh.xdmf"), "output_path": str(tmp_path / "results")}
    config_path = tmp_path / "config.toml"

    config_path.write_text(toml.dumps(config))
    assert main([str(config_path)]) == 0
    assert (tmp_path / "results/output.txt").is_file()
    assert (tmp_path / "results/output.txt").read_text() == "Mesh has 216 vertices."
