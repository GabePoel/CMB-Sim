# CMB Sim

Create early universe models and see how they evolve! Then, render maps of the background radiation for each universe you make!

![Oscillating Universe](Figures/Periodic.gif)

## Pre-Run Simulations

The evolution of several different types of universes are already generated! [The data sets](Data%20Sets) and [figures with additional information](Figures) are available in this repository. But, I encourage you to watch the videos in the table below.

| Universe                                                     | Number of Particles | Number of Time Steps |
| ------------------------------------------------------------ | ------------------- | -------------------- |
| [Slow logarithmically shrinking universe](https://youtu.be/zcKA7e664Zw) | 100                 | 1000                 |
| [Slow exponentially growing universe](https://youtu.be/HDdtpSWe0bA) | 1000                | 1000                 |
| [Logarithmically shrinking then growing universe](https://youtu.be/L5UMsxqhVtE) | 100                 | 5000                 |
| [Exponentially growing universe](https://youtu.be/IZ74dyzGOCs) | 100                 | 1000                 |
| [Oscillating universe](https://youtu.be/dCtzyYYSN6Y)         | 1000                | 300                  |
| [Tiny static universe](https://www.youtube.com/watch?v=_x_g3oanCP8) | 1000                | 300                  |
| [Slightly larger static universe](https://youtu.be/FiM_chzzZQg) | 1000                | 100                  |

The additional figures provide the composition of each universe, size, number of Thomson scattering events, number of recombination events, and number of ionization events as a function of time. An example set of these are displayed for the slow logarithmically shrinking universe down below.

## How CMB-Sim Works

### The Evolution Procedure

This is the back end for how CMB-Sim creates and evolves universes. The code for this is all in [`cmb_sim.py`](cmb_sim.py).

1. A new universe is created. It is defined to be geometrically flat and cubic with periodic boundary conditions. This simplifies the scale transformations and allows each particle to have many interactions even when a relatively small number of particles is simulated. The universe’s “size” refers to the length of one of its edges in natural units.
2. The universe gets populated with particles. In all the pre-run simulations, these start out as half protons and half electrons. However, the makeup of the universe can change as it evolves with time. CMB-Sim universes have only four types of particles: electrons, protons, photons, and hydrogen atoms.
3. The universe evolves! This requires specifying step size Δt and some function which determines how the size of the universe changes with each time step. Within each step of evolution, the following procedure occurs:
    1. The electric and gravitational fields at each particle are calculated. This is done non-relativistically. The program assumes Newtonian gravity and does not determine values for the magnetic field.
    2. Calculate the force on each particle due to the fields at its location. Then, update the particles coordinates in state space in accordance with the forces acting on them. All updated velocities are calculated relativistically using Einstein’s velocity addition rule. Note that this is irrelevant for photons since they are represented with no mass nor charge and the fields are non-relativistic. Photon interactions are determined in later steps.
        ![Equation 1](Equations/eq1.svg)
    3. Determine if any photons are within the Thomson cross section of any other particles. Proceed to scatter them! This is done for all baryonic matter and not just free electrons. However, the effect is still dominated by interactions with free electrons. The default cross sectional formula is shown below. Note that it’s further simplified in the code since CMB-Sim works with natural units. Given the computational difficulty of simulating interactions for a large number of particles and the relatively small size of the Thomson cross section, CMB-Sim gives the option of using expanded cross sections. This ensures that interactions still happen with an otherwise sparsely populated universe.
        ![Equation 2](Equations/eq2.svg)
    4. The temperature of the universe is calculated. We assume that the universe is in thermal equilibrium and that radiation pressure dominates. If <E> is the mean energy of a photon, the temperature (assuming only the constituents in the CMB Sim model) ultimately ends up simplifying to the following expression. Note that recombination to hydrogen atoms is the only interaction that produces new photons in CMB-Sim. As such, there is not a wide spread of photon energies to average over.
        ![Equation 2](Equations/eq3.svg)
    5. Determine if any electrons are within the Bohr radius of any protons. When this is the case, determine the deBroglie wavelength and local electron density. Then apply the Saha equation to determine the expected ratio of protons to hydrogen atoms. This is transformed to the probability by which any sufficiently close electron-proton pairs recombine. A similar process is used for any sufficiently dense photon-hydrogen pairs. Note that each particle is only permitted to undergo one interaction in every time step. This means any repeated recombination-ionization cycles of nearby particles must manifest in the data over a period of time. It also puts a computational limit on these interaction rates. In all interactions, energy and momentum is conserved. However, newly produced particles are projected out along a random directions.
    6. Once all interactions are complete, scale the universe to its new size for the next time step. Update the positions, velocities, and redshifts of all particles accordingly.

### Radiation Maps

This is how the actual pretty radiation maps are made! The code for this is all in [`cmb_processing.ipynb`](cmb_processing.ipynb). If you have `ffmpeg` installed, this also allows for creating videos like those in the pre-run simulations.

For creating radiation maps, the “Earth” is assumed to be at the “center” of the universe. Given the periodic box coordinate system used in CMB-Sim, this is well defined. The photons along the last scattering edge relative to the universe’s center are projected to the surface of an infinitesimal sphere surrounding the center point. Each photon gets its redshift determined one last time given its state when the radiation map is calculated. This takes into account the redshift due to their comoving velocity as well as due to the expansion rate of the universe. Both transverse and radial redshifts are determined. The final redshift is used to determine the wavelength of the photon as seen from the center coordinate. The wavelengths for all photons are then normalized to the visual spectrum.

Since each photon is a single discretized particle, this means that the background radiation is only determined at a handful of points on the projected sphere. To make a complete (and pretty!) radiation map, the space between these points is linearly interpolated over to create a continuous meshgrid. The final spherical surface is then displayed in a Mollweide projection.

## Example: Slow Logarithmically Shrinking Universe

CMB Sim saves the complete state of each simulated universe during each time step. This allows for the data of each universe to be processed afterwards. To show what this looks like, here’s several quantities calculated for a [slow logarithmically shrinking example universe](https://www.youtube.com/watch?v=zcKA7e664Zw).

### Universe Size

This universe is one that begins quite large and then undergoes a slow logarithmic shrinking. This may not be an especially “physical” universe. However, it is a helpful toy model of what happens to a universe as it slowly compresses down.

![Slow Logarithmic Size - Linear](Figures/Slow%20Logarithmic%20Size%20-%20Linear.png)

### Universe Composition

![Universe Composition](Figures/Slow%20Logarithmic%20Out.png)

As the universe shrinks, it universe becomes hot and highly ionized. We can even see the exact moments and rate at which ionization occurs.

![Universe Composition](Figures/Slow%20Logarithmic%20Ionization.png)

### Universe Opacity

Once the universe gets especially small, it is already highly ionization and is sufficiently saturated to prevent further events. However, this has resulted in a large number of free electrons for the tightly packed photons to scatter off of. We can see the high rate at which photons are scattering reflected in the number of Thomson scattering events that occur. Note that the spike at the final time step is spurious since that’s when the universe collapses down to a single point.

![Slow Logarithmic Thomson](Figures/Slow%20Logarithmic%20Thomson.png)

 In the late stages of this universe, very few photons remained. So, at these small size scales, we can see that nearly every photon scattered during each time step! Hence, we can clearly see the regime where this universe has become opaque.

## Notes on Running CMB Sim

CMB Sim needs to wrapper in order to call it, instantiate universes with the desired parameters, and create all the time evolution data. Several example scripts are provided to create a [logarithmic](Data%20Sets/logarithmic_run.py), [oscillating](Data%20Sets/periodic_run.py), or [static](Data%20Sets/tiny_run.py) universe. Please see the docstrings in [`cmb_sim.py`](cmb_sim.py) for information on how to create/manipulate a universe and assign it the correct multiprocessing flags. Rather than creating a new universe from scratch, you can also mess around with the [pre-existing data sets](Data%20Sets). These have many files and tend to be quite large. But, they can be played with directly from [`cmb_processing.ipynb`](cmb_processing.ipynb).
