# simcardems2

Simcardems2 is the next version of [Simula Cardiac Electro-Mechanics Solver (simcardems)](https://github.com/ComputationalPhysiology/simcardems) and a part of the [SimCardioTest](https://www.simcardiotest.eu/wordpress/) project. The solver uses [pulse](https://github.com/finsberg/pulse) and [fenics-beat](https://github.com/finsberg/fenics-beat) to solve mechanics and electrophysiology, respectively. Both libraries are based on [FEniCS](https://fenicsproject.org/).

# Install with Docker
To run simcardems2 in a Docker container, you can use the following [pre-built Docker image](https://github.com/scientificcomputing/packages/pkgs/container/fenics-gmsh) which has the necessary dependancies, including FEniCS:


```console
docker pull ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
```


