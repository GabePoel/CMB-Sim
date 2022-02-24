# CMB Sim

Generate early universe models and simulate the evolution of their content. Then, generate the background radiation for each universe you make!

![Oscillating Universe](Figures/Periodic.gif)

## Pre-Run Simulations

The evolution of several different types of universes are already generated! [The data sets](Data%20Sets) and [figures with additional information](Figures) are available in this repository. But, I encourage you to watch the videos below:

| Universe                                                     | Number of Particles | Number of Time Steps |
| ------------------------------------------------------------ | ------------------- | -------------------- |
| [Slow logarithmically shrinking universe](https://youtu.be/zcKA7e664Zw) | 100                 | 1000                 |
| [Slow exponentially growing universe](https://youtu.be/HDdtpSWe0bA) | 1000                | 1000                 |
| [Logarithmically shrinking then growing universe](https://youtu.be/L5UMsxqhVtE) | 100                 | 5000                 |
| [Exponentially growing universe](https://youtu.be/IZ74dyzGOCs) | 100                 | 1000                 |
| [Oscillating universe](https://youtu.be/dCtzyYYSN6Y)         | 1000                | 300                  |
| [Tiny static universe](https://www.youtube.com/watch?v=_x_g3oanCP8) | 1000                | 300                  |
| [Slightly larger static universe](https://youtu.be/FiM_chzzZQg) | 1000                | 100                  |

The additional figures provide the composition of each universe, size, number of Thomson scattering events, number of recombination events, and number of ionization events as a function of time.

## The Simulation Procedure

1. A new universe is created. It is defined to be geometrically flat and cubic with periodic boundary conditions. This simplifies the scale transformations and allows each particle to have many interactions even when a relatively small number of particles is simulated. The universe’s “size” refers to the length of one of its edges in natural units.
2. The universe gets populated with particles. In all the pre-run simulations, these start out as half protons and half electrons. However, the makeup of the universe can change as it evolves with time. CMB-Sim universes have only four types of particles: electrons, protons, photons, and hydrogen atoms.
3. The universe evolves! This requires specifying step size Δt and some function which determines how the size of the universe changes with each time step. Within each step of evolution, the following procedure occurs:
    1. The electric and gravitational fields at each particle are calculated. This is done non-relativistically. The program assumes Newtonian gravity and does not determine values for the magnetic field.
    2. Calculate the force on each particle due to the fields at its location. Then, update the particles coordinates in state space in accordance with the forces acting on them. All updated velocities are calculated relativistically using Einstein’s velocity addition rule. Note that this is irrelevant for photons since they are represented with no mass nor charge and the fields are non-relativistic. Photon interactions are determined in later steps.
        ![Equation 1](Equations/eq1.svg)
    3. Determine if any photons are within the Thomson cross section of any other particles. This is done for all baryonic matter and not just free electrons. However, the effect is still dominated by interactions with free electrons. The default cross sectional formula is shown below. Note that it’s further simplified in the code since CMB-Sim works with natural units. Given the computational difficulty of simulating interactions for a large number of particles and the relatively small size of the Thomson cross section, CMB-Sim gives the option of using expanded cross sections. This ensures that interactions still happen with an otherwise sparsely populated universe.
        ![Equation 2](Equations/eq2.svg)
    4. 

### Thomson Scattering

### Recombination

### Ionization

## Notes for Running CMB-Sim
