# simcardems2

Simcardems2 is the next version of [Simula Cardiac Electro-Mechanics Solver (simcardems)](https://github.com/ComputationalPhysiology/simcardems) and a part of the [SimCardioTest](https://www.simcardiotest.eu/wordpress/) project. The solver uses [pulse](https://github.com/finsberg/pulse) and [fenics-beat](https://github.com/finsberg/fenics-beat) to solve mechanics and electrophysiology, respectively. Both libraries are based on [FEniCS](https://fenicsproject.org/).

# Install with Docker
Create a directory for simcardems and clone the github repository:
to be shared with your Docker container:
```shell
mkdir simcardems2_share
cd simcardems2_share
gh repo clone ComputationalPhysiology/simcardems2
```

To run simcardems2 in a Docker container, you can use the following [pre-built Docker image](https://github.com/scientificcomputing/packages/pkgs/container/fenics-gmsh) which includes the FEniCS library:

```shell
docker pull ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
```

Inside your simcardems2 directory, create a Docker container, here called "simcardems2", using the Docker image provided above:

```shell
docker run --name simcardems2 -v "$(pwd)":/shared -w /shared -it ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
```
The option -v mounts the current directory and its files to the new container.


Inside the Docker container, enter the simcardems2 folder and install simcardems2:

```shell
cd simcardems2/
python3 -m pip install -e .
```
