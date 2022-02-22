#!/usr/bin/python3

import numpy as np
import multiprocessing as mp
from scipy.special import zeta

"""
Units
-----

These are the basic constant units used throughout the universe generation and
CMB simulation.
"""
k = 1
c = 1
G = 0.0001 # Gravity is scaled to be much smaller than the electric field.
vcut = 0.9 * c
# G = 0
hbar = 1
electric_constant = 1
electron_mass = 1
bohr_radius = 1
alpha = 1 / 137 # Comment if pure natural units.
# alpha = e^2 / (4 * np.pi) Uncomment if pure natural units.
hydrogen_mass = 1837.5 * electron_mass
proton_mass = 1836.15267 * electron_mass
photon_mass = hydrogen_mass - proton_mass - electron_mass
photon_energy = np.abs(proton_mass + electron_mass - hydrogen_mass) * c ** 2
interaction_rate = 0.5
interaction_radius = 0.1


def vel_add(v1, v2):
    """
    Add two velocities together relativistically. This assumes hyperbolic
    spacetime at caps at the defined speed of light.

    Parameters
    ----------
    v1 : array
        Three element array specifying some velocity in Cartesian coordinates.
    v2 : array
        Three element array specifying some velocity in Cartesian coordinates.

    Returns
    -------
    array
        Three element array specifying the summed velocity in Cartesian
        coordinates.
    """
    return (v1 + v2) / (1 + (v1 * v2 / c ** 2))


def dist(p1, p2):
    """
    Find cartesian distance between two particles modulo the size of the
    universe. There is a minimum assumed distance of 0.001 in order to prevent
    singularities or numerical floating point errors.

    Parameters
    ----------
    p1 : Particle
        One of the two particles. This is the particle that the universe is
        detected from.
    p2 : Particle
        The other particle.

    Returns
    -------
    float
        Distance between the particles.
    """
    size = p1.universe.size
    translations = size * np.array([[0, 0, 0],
                                    [1, 0, 0],
                                    [-1, 0, 0],
                                    [0, 1, 0],
                                    [0, -1, 0],
                                    [0, 0, 1],
                                    [0, 0, -1]])
    dists = []
    for t in translations:
        d = np.sqrt(np.sum((p1.x - (p2.x + t)) ** 2))
        dists.append(d)
    return max(min(dists), 0.001)


def redshift(particle, last_universe, time_step):
    """
    Returns z + 1

    Parameters
    ----------
    particle : Particle
        The particle in question.
    last_universe : Universe
        Universe of the previous time step.
    time_step : float
        Length of time step.

    Returns
    -------
    float
        z + 1
    """
    v0 = particle.v
    a = particle.universe.size
    a_old = last_universe.size
    x = particle.x - np.ones((3)) * a / 2
    v = (a - a_old) * x / time_step + a * v0
    [vx, vy, vz] = list(v)
    vtheta = np.arctan(np.sqrt(vx ** 2 + vy ** 2) / vz)
    vphi = np.arctan(vy / vx)
    rhat = np.array([np.sin(vtheta) * np.cos(vtheta),
                     np.sin(vtheta) * np.sin(vtheta),
                     np.cos(vtheta)])
    v_rad = np.dot(v, rhat)
    v_trn = np.sqrt(np.sqrt(np.sum(v ** 2)) - v_rad ** 2)
    z_rad = np.sqrt((1 + v_rad / 1) / (1 - v_rad / 1)) - 1
    z_trn = 1 / np.sqrt(1 - v_trn ** 2 / 1) - 1
    return np.sqrt(z_rad ** 2 + z_trn ** 2) + 1


def wavelength(redshift, units=True):
    """
    Return the wavelength of the photon given its redshift.

    Parameters
    ----------
    redshift : float
        z + 1
    units : bool, optional
        Whether to include real units, by default True.

    Returns
    -------
    float
        The wavelength in nanometers.
    """
    emit_energy = 13.6
    wl = (1 + redshift) * 2 * np.pi / emit_energy
    if units:
        return 197.3 * wl


def scale_wavelengths(arr):
    """
    Normalizes wavelengths to the visible range.

    Parameters
    ----------
    arr : array
        Array of floats representing the wavelengths for many particles.

    Returns
    -------
    arr
        Array of floats representing the normalized wavelengths for many
        particles.
    """
    min_wl = min(arr)
    max_wl = max(arr)
    base_wl = (arr - min_wl) / (max_wl - min_wl)
    new_wl = 380 + base_wl * (750 - 380)
    return new_wl


def wavelength_to_rgb(wavelength):
    """
    Convert visible wavelengths to the corresponding RGB values.

    Parameters
    ----------
    wavelength : float
        A visible light wavelength in nanometers.

    Returns
    -------
    list
        Three element list of the red, green, and blue values normalized to be
        between zero and one. This is done instead of scaling them out of 255
        so that it can be further manipulated by the color conversion tools in
        matplotlib.
    """
    w = int(wavelength)

    # color
    if w >= 380 and w < 440:
        R = -(w - 440.) / (440. - 350.)
        G = 0.0
        B = 1.0
    elif w >= 440 and w < 490:
        R = 0.0
        G = (w - 440.) / (490. - 440.)
        B = 1.0
    elif w >= 490 and w < 510:
        R = 0.0
        G = 1.0
        B = -(w - 510.) / (510. - 490.)
    elif w >= 510 and w < 580:
        R = (w - 510.) / (580. - 510.)
        G = 1.0
        B = 0.0
    elif w >= 580 and w < 645:
        R = 1.0
        G = -(w - 645.) / (645. - 580.)
        B = 0.0
    elif w >= 645 and w <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    # intensity correction
    if w >= 380 and w < 420:
        SSS = 0.3 + 0.7 * (w - 350) / (420 - 350)
    elif w >= 420 and w <= 700:
        SSS = 1.0
    elif w > 700 and w <= 780:
        SSS = 0.3 + 0.7 * (780 - w) / (780 - 700)
    else:
        SSS = 0.0
    SSS *= 255

    return np.array([int(SSS * R), int(SSS * G), int(SSS * B)]) / 255


def check_types(particles, types):
    """
    Make sure the provided particles are of the intended types.

    Parameters
    ----------
    particles : list
        List of particles to check.
    types : list
        List of strings for the particle types.

    Returns
    -------
    bool
        Whether or not all the particles are within the allowed tyupes.
    """
    included = [p.type for p in particles]
    return all([t in included for t in types])


def sort_types(particles, types):
    """
    Sort a list of particles by the order specified in a list of types. Any
    particles of types not in the provided types list will be placed at the
    end of the output list.

    Parameters
    ----------
    particles : list
        List of particles to sort.
    types : list
        List of strings of particle types in the order you want the particles
        outputted.

    Returns
    -------
    list
        Sorted list of particles.
    """
    sorted_particles = []
    for t in types:
        for p in particles:
            if p.type == t and p not in sorted_particles:
                sorted_particles.append(p)
    for p in particles:
        if p not in sorted_particles:
            sorted_particles.append(p)
    return sorted_particles


def thomson_radius(particle, increase=True):
    """
    Get the radius of the Thomson cross section for the provided particle.

    Parameters
    ----------
    particle : Particle
        The particle to consider.
    increase : bool, optional
        Increase the size of the Thomson cross section in a slightly unphysical
        way. This defaults to True. This is done because of the small number of
        particles in these simulated universes compared to those of a real
        universe. In order to make sure that any scattering happens at all we
        either need to make the universe always be so small that we have to
        start worrying about floating point error or we just increase the size
        of the Thomson cross section.

    Returns
    -------
    float
        Radius of the Thomson cross section.
    """
    if particle.type == 'photon':
        return 0
    elif particle.type == 'electron' and increase:
        return 0.05
    m = particle.m
    return 8 * np.pi / 3 * (alpha * hbar * c / (m * c ** 2)) ** 2


def smoother(x):
    """
    Cutoff function for smoothly truncating near the speed of light.

    Parameters
    ----------
    x : arraylike
        Float or array of floats.

    Returns
    -------
    arraylike
        Float or array of floats.
    """
    return -1 / (x + 1) + 1


def truncate_velocity(v):
    """
    Emergency velocity cutoff tool for when the discontinuity of continuous
    time steps and operation ordering allows for non-photon particles to
    accidentally exceed the speed of light. Note that they only do this until
    they are corrected. Hence, it never manifests in any of the physics of the
    simulation although it may appear as spurious points in the data due to
    when snapshots are captured.

    Parameters
    ----------
    v : array
        Three element array corresponding to the nonrelativistic velocity of
        some particle.

    Returns
    -------
    array
        Three element array corresponding to the truncated nonrelativistic
        velocity of some particle.
    """
    mag = np.sqrt(np.sum(v) ** 2)
    if mag > vcut:
        new_mag = (1 - vcut) * smoother((mag - vcut) / (1 - vcut)) + vcut
        v = new_mag * v / mag
    return v


def random_direction():
    """
    Generate a randomly oriented normal vector in Cartesian coordinates.

    Returns
    -------
    array
        The three element normal vector.
    """
    phi = np.random.random() * 2 * np.pi
    theta = np.random.random() * np.pi
    x = c * np.sin(theta) * np.cos(phi)
    y = c * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    return np.array([x, y, z])


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis : array
        Normal vector pointing along the axis of rotation.
    theta : float
        Angle with which to rotate.

    Returns
    -------
    array
        Rotation matrix to make the corresponding transformation.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def get_cmb(u):
    """
    Project CMB onto surface of a sphere.

    Parameters
    ----------
    u : Universe

    Returns
    -------
    arr
        Cartesian coordinates of CMB photons on a sphere with radius one.
    """
    x = []
    y = []
    z = []
    offset = np.ones((3)) * u.size / 2
    for p in u.particles:
        if p.type == 'photon':
            [x0, y0, z0] = p.x - offset
            mag = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2)
            x.append(x0 / mag)
            y.append(y0 / mag)
            z.append(z0 / mag)
    return np.array([x, y, z])


class MultiprocessHelper:
    def __init__(self, var, caller):
        self.var = var
        self.caller = caller

    def make_caller(self):
        return self.caller(self.var)


class Evolver:
    """
    Evolve particle in time due to the forces on it. Helper for
    multiprocessing.
    """

    def __init__(self, time):
        self.time = time

    def __call__(self, particle):
        particle.evolve(self.time)
        particle.G = np.zeros((3))
        particle.E = np.zeros((3))


class Graviter:
    """
    Determine gravitational field at site of particle. Helper for
    multiprocessing.
    """

    def __init__(self, var):
        """
        Initialize the gravity helper.

        Parameters
        ----------
        var : None
            Dummy variable.
        """
        pass

    def __call__(self, particles):
        """
        Determine gravitational field due to the interaction of two particles.

        Parameters
        ----------
        particles : list
            Two element list of particles. The first one is experiencing the
            gravitational field due to interactions with the second.
        """
        p1, p2 = particles[0], particles[1]
        if p1 is not p2:
            G0 = -G * p2.m * (p1.x - p2.x) / dist(p1, p2) ** 3
            p1.G += G0


class Electricer:
    """
    Determine electric field at site of particle. Helper for multiprocessing.
    """

    def __init__(self, var):
        """
        Initialize the electric field helper.

        Parameters
        ----------
        var : None
            Dummy variable.
        """
        pass

    def __call__(self, particles):
        """
        Determine the electric field due to the interaction of two particles.

        Parameters
        ----------
        particles : list
            Two element list of particles. The first one is experiencing the
            electric field due to interactions with with second.

        Returns
        -------
        array
            Three element array of the force of the electric field.
        """
        p1, p2 = particles[0], particles[1]
        if p1 is not p2:
            E0 = 1 / (4 * np.pi * electric_constant) * \
                p2.q * (p1.x - p2.x) / dist(p1, p2) ** 3
        else:
            E0 = np.zeros((3))
        return E0


class Interacter:
    """
    Generate scattering interactions between particles. Helper for
    multiprocessing.
    """

    def __init__(self, var):
        """
        Initialize the interaction helper.

        Parameters
        ----------
        var : None
            Dummy variable.
        """
        pass

    def __call__(self, particles):
        """
        Determine the dominant interaction between two particles.

        Parameters
        ----------
        particles : list
            Two element list of particles. Both of them are affected by any
            interactions that may occur.
        """
        p1, p2 = particles[0], particles[1]
        if p1 is not p2:
            p1.universe.thomson(p1, p2)
            c1 = np.random.random() <= interaction_rate
            c2 = dist(p1, p2) <= interaction_radius
            if c1 and c2:
                p1.universe.ionization(p1, p2)
                p1.universe.recombination(p1, p2)


class Measurer:
    """
    Measure the distance between two particles. Helper for multiprocessing.
    """

    def __init__(self, var):
        """
        Initialize the measurement helper.

        Parameters
        ----------
        var : None
            Dummy variable.
        """
        pass

    def __call__(self, particles):
        """
        Determine the distance between two particles.

        Parameters
        ----------
        particles : list
            Two element list of particles.

        Returns
        -------
        float
            The distance between the two particles. Note that this is measured
            in Minkowski space.
        """
        p1, p2 = particles[0], particles[1]
        return (p1, p2, dist(p1, p2))


class Universe:
    """
    The `Universe` class contains all data for the universe at a given point in
    time. This includes the size of the universe, its temperature, and the
    information for every particle contained within it. All the deterministic
    physics at a universe in a later time evolution can be derived from an
    earlier universe. In order to represent the complete time evolution of a
    universe, have a list of instances of the `Universe` class.
    """

    def __init__(
            self,
            size=1,
            workers=1,
            mpG=False,
            mpE=False,
            mpEvolve=True,
            mpInteract=False,
            mpMeasure=False,
            import_set=None):
        """
        Initiate an instance of the `Universe` class.

        Parameters
        ----------
        size : int, optional
            Size of the universe in natural units, by default 1.
        workers : int, optional
            Number of processing cores to use when calculating the physics of
            the universe, by default 1.
        mpG : bool, optional
            Whether or not to use multiprocessing when calculating the
            gravitational field in the universe. False by default.
        mpE : bool, optional
            Whether or not to use multiprocessing when calculating the
            electric field in the universe. False by default.
        mpEvolve : bool, optional
            Whether or not to use multiprocessing when calculating the
            evolution of particles in the universe. True by default.
        mpInteract : bool, optional
            Whether or not to use multiprocessing when calculating the
            scattering interactions of particles in the universe. False by
            default.
        mpMeasure : bool, optional
            Whether or not to use multiprocessing when calculating the
            distance between particles in the universe. False by default.
        import_set : set, optional
            Universes can be exported as sets and saved as json files using the
            `to_json` method. When a set of a loaded universe json is passed
            into a new universe, it will import the data and create a clone
            of the provided universe. This allows for the saving and sharing of
            data. When no such universe set is provided, this defaults to None
            and a new universe is created from scratch with no particles or
            other imported parameters in it.
        """
        self.size = size
        self.workers = workers
        self.mpG = mpG
        self.mpE = mpE
        self.mpEvolve = mpEvolve
        self.mpInteract = mpInteract
        self.mpMeasure = mpMeasure
        self.particles = []
        self.T = None
        self.next_size = self.size
        self.proton_prob = None
        if import_set is not None:
            for particle_name in import_set['particles'].keys():
                particle_set = import_set['particles'][particle_name]
                self.import_particle(particle_set, particle_name)
            self.size = import_set['size']

    def __repr__(self):
        """
        String representation of the `Universe` class.

        Returns
        -------
        str
            A string describing this universe.
        """
        s = 'Size: ' + str(self.size)
        s += '\nT: ' + str(self.T)
        for particle in self.particles:
            s += '\n\n' + str(particle)
        return s

    def import_particle(self, import_set, name):
        """
        Import a particular particle from the provided set. These sets are
        gathered json files corresponding to either an individual particle or
        a universe of them. Also see the corresponding methods in the
        `Particle` class for more details. Each imported with this method
        defines only one particle. And each particle imported in this manner
        will be directly added to this universe.

        Parameters
        ----------
        import_set : set
            Set describing all the parameters for a particle. These are usually
            imported from a json file.
        name : str
            Name of the provided particle. This is not typically saved within
            the json file for an individual particle. However, they are used
            as the dictionary keys to identify particles saved within the json
            file for a universe.
        """
        p = Particle(self)
        p.type = import_set['type']
        p.m = import_set['m']
        p.q = import_set['q']
        p.x = np.array(import_set['x'])
        p.v = np.array(import_set['v'])
        p.last_interaction = import_set['last interaction']
        p.name = name

    def interface_func(self, f, x):
        """
        Apply a called function from a multiprocessing helper class to the
        provided objects.

        Parameters
        ----------
        f : function
            Function called from a multiprocessing helper.
        x : list
            List of some objects to run the function on.

        Returns
        -------
        list
            List of objects outputted by the function.
        """
        pool = mp.Pool(self.workers)
        return list(pool.map(f, x))

    def evolve(self, time):
        """
        Evolve the position and velocity of all the particles in this universe.

        Parameters
        ----------
        time : float
            Time step to evolve by. This is assumed to be in natural units.
        """
        self.stage_E()
        self.stage_G()
        if self.mpEvolve:
            evolver = MultiprocessHelper(time, Evolver).make_caller()
            self.interface_func(evolver, self.particles)
        else:
            for p in self.particles:
                p.evolve(time)
                p.G = np.zeros((3))
                p.E = np.zeros((3))
        self.stage_proton_prob()
        self.stage_interactions()
        for p in self.particles:
            p.interacted = False
            p.x = self.next_size / self.size * p.x
            p.v = truncate_velocity(p.v)
        self.size = self.next_size

    def stage_interactions(self):
        """
        Stage all the scattering interactions that could occur between
        particles in this universe.
        """
        if self.mpMeasure:
            p_table = []
            for p1 in self.particles:
                for p2 in self.particles:
                    p_table.append((p1, p2))
            measurer = MultiprocessHelper(1, Measurer).make_caller()
            ps = self.interface_func(measurer, p_table)
            print(ps)
        else:
            ps = []
            for i in range(len(self.particles)):
                for j in range(len(self.particles)):
                    d = dist(self.particles[i], self.particles[j])
                    ps.append((self.particles[i], self.particles[j], d))
        ps.sort(key=lambda t: t[2])
        if self.mpInteract:
            interacter = MultiprocessHelper(1, Interacter).make_caller()
            self.interface_func(interacter, ps)
        else:
            for pair in ps:
                if pair[0] is not pair[1]:
                    self.thomson(pair[0], pair[1])
                    if np.random.random() <= interaction_rate and dist(
                            pair[0], pair[1]) <= interaction_radius:
                        self.ionization(pair[0], pair[1])
                        self.recombination(pair[0], pair[1])

    def stage_E(self):
        """
        Stage the values of the electric field for each particle.
        """
        if self.mpE:
            p_table = []
            for p1 in self.particles:
                for p2 in self.particles:
                    p_table.append((p1, p2))
            electricer = MultiprocessHelper(1, Electricer).make_caller()
            E_table = np.array(self.interface_func(electricer, p_table))
            E_table.reshape((len(self.particles), len(self.particles), 3))
            print(
                E_table.reshape(
                    (len(
                        self.particles), len(
                        self.particles), 3)))
            # Multiprocessing is broken :(
        else:
            for p1 in self.particles:
                for p2 in self.particles:
                    if p1 is not p2:
                        E0 = 1 / (4 * np.pi * electric_constant) * \
                            p2.q * (p1.x - p2.x) / dist(p1, p2) ** 3
                        p1.E += E0

    def stage_B(self):
        """
        Stage the values of the magnetic field for each particle. Not currently
        implemented.
        """
        pass

    def stage_G(self):
        """
        Stage the values of the gravitational field for each particle.
        """
        if self.mpG:
            p_table = []
            for p1 in self.particles:
                for p2 in self.particles:
                    p_table.append((p1, p2))
            graviter = MultiprocessHelper(1, Graviter).make_caller()
            self.interface_func(graviter, p_table)
        else:
            for p1 in self.particles:
                for p2 in self.particles:
                    if p1 is not p2:
                        G0 = -G * p2.m * (p1.x - p2.x) / dist(p1, p2) ** 3
                        p1.G += G0

    def stage_T(self):
        """
        Stage the temperature of the universe. We assume that it's in thermal
        equilibrium. This is calculated only for radiation pressure.
        """
        self.T = zeta(3) / zeta(4) * photon_energy / 3 / k

    def stage_proton_prob(self):
        """
        Set the ratio of protons to neutral hydrogen atoms via the Saha
        equation.
        """
        self.stage_T()
        debroglie_wavelength = np.sqrt(
            2 * np.pi * hbar / (electron_mass * k * self.T))
        electron_density = self.electron_count() / self.size ** 3
        num = np.exp(-photon_energy / (k * self.T))
        den = electron_density * debroglie_wavelength ** 3
        if den <= 0:
            self.proton_prob = 1
        else:
            proton_ratio = num / den
            self.proton_prob = proton_ratio / (proton_ratio + 1)

    def photon_count(self):
        """
        Get the number of photons in this universe.

        Returns
        -------
        int
            Number of photons.
        """
        return np.sum([1 for p in self.particles if p.type == 'photon'])

    def electron_count(self):
        """
        Get the number of electrons in this universe.

        Returns
        -------
        int
            Number of electrons.
        """
        return np.sum([1 for p in self.particles if p.type == 'electron'])

    def proton_count(self):
        """
        Get the number of protons in this universe.

        Returns
        -------
        int
            Number of protons.
        """
        return np.sum([1 for p in self.particles if p.type == 'proton'])

    def hydrogen_count(self):
        """
        Get the number of hydrogen atoms in this universe.

        Returns
        -------
        int
            Number of hydrogen atoms.
        """
        return np.sum([1 for p in self.particles if p.type == 'hydrogen'])

    def thomson(self, p1, p2):
        """
        Attempt Thomson scattering between two interacting particles.

        Parameters
        ----------
        p1 : Particle
            One of the two interacting particles
        p2 : Particle
            One of the two interacting particles.
        """
        [ph, po] = sort_types([p1, p2], 'photon')
        c1 = ph.type == 'photon'
        c2 = po.type != 'photon'
        c3 = dist(p1, p2) <= thomson_radius(po)
        c4 = True
        c5 = True
        if all([c1, c2, c3, c4, c5]):
            n = ph.x / np.sqrt(np.sum(ph.x ** 2))
            r = (ph.x - po.x) / np.sqrt(np.sum((ph.x - po.x) ** 2))
            theta = np.arccos(np.dot(n, r))
            photon_momentum = ph.v * photon_mass
            other_momentum = po.v * po.m
            total_momentum = photon_momentum + other_momentum
            plane = np.cross(n, r)
            plane = plane / np.abs(plane)
            R = rotation_matrix(plane, theta)
            ph.v = np.matmul(R, ph.v)
            photon_momentum = ph.v * photon_mass
            other_momentum = total_momentum - photon_momentum
            po.v = other_momentum / po.m
            ph.interacted = True
            po.interacted = True
            ph.last_interaction = 'Thomson with ' + str(po.name)
            po.last_interaction = 'Thomson with ' + str(ph.name)

    def ionization(self, p1, p2):
        """
        Attempt ionization between two interacting particles. This is only done
        between photons and hydrogen.

        Parameters
        ----------
        p1 : Particle
            One of the two interacting particles.
        p2 : Particle
            One of the two interacting particeles.
        """
        [pH, ph] = sort_types([p1, p2], ['hydrogen', 'photon'])
        c1 = check_types([p1, p2], ['hydrogen', 'photon'])
        c2 = not pH.interacted
        c3 = not ph.interacted
        c4 = np.random.random() <= self.proton_prob
        if all([c1, c2, c3, c4]):
            direction = random_direction()
            photon_momentum = ph.v * photon_mass
            total_momentum = photon_momentum + pH.m * pH.v
            electron_ratio = np.random.random()
            electron_momentum = electron_ratio * direction * \
                np.sqrt(np.sum(total_momentum ** 2))
            proton_momentum = total_momentum - electron_momentum
            pH.to_proton()
            ph.to_electron()
            pH.v = proton_momentum / pH.m
            ph.v = electron_momentum / ph.m
            pH.interacted = True
            ph.interacted = True
            pH.last_interaction = 'Ionization with ' + str(ph.name)
            ph.last_interaction = 'Ionization with ' + str(pH.name)

    def recombination(self, p1, p2):
        """
        Attempt recombination between two interacting particles. This is only
        done between protons and electrons.

        Parameters
        ----------
        p1 : Particle
            One of the two interacting particles.
        p2 : Particle
            One of the two interacting particles.
        """
        [pp, pe] = sort_types([p1, p2], ['proton', 'electron'])
        c1 = check_types([p1, p2], ['proton', 'electron'])
        c2 = not pp.interacted
        c3 = not pe.interacted
        c4 = np.random.random() > self.proton_prob
        if all([c1, c2, c3, c4]):
            direction = random_direction()
            photon_velocity = c * direction
            photon_momentum = photon_velocity * photon_mass
            total_momentum = pp.m * pp.v + pe.v * pe.v
            hydrogen_momentum = total_momentum - photon_momentum
            pp.to_hydrogen()
            pe.to_photon()
            pp.v = hydrogen_momentum / pp.m
            pe.v = photon_velocity
            pp.interacted = True
            pe.interacted = True
            pp.last_interaction = 'Recombination with ' + str(pe.name)
            pe.last_interaction = 'Recombination with ' + str(pp.name)

    def add_electrons(self, count=1):
        """
        Add some number of randomly distributed electrons to this universe.
        Electrons start out with no velocity.

        Parameters
        ----------
        count : int, optional
            How many electrons to add, by default 1.
        """
        for i in range(count):
            p = Particle(self)
            p.to_electron()
            p.random_teleport()

    def add_protons(self, count=1):
        """
        Add some number of randomly distributed protons to this universe.
        Protons start out with no velocity.

        Parameters
        ----------
        count : int, optional
            How many protons to add, by default 1.
        """
        for i in range(count):
            p = Particle(self)
            p.to_proton()
            p.random_teleport()

    def add_hydrogens(self, count=1):
        """
        Add some number of randomly distributed hydrogen atoms to this
        universe. Hydrogen atoms start out with no velocity.

        Parameters
        ----------
        count : int, optional
            How many hydrogen atoms to add, by default 1.
        """
        for i in range(count):
            p = Particle(self)
            p.to_hydrogen()
            p.random_teleport()

    def add_photons(self, count=1):
        """
        Add some number of randomly distributed photons to this universe.
        Photons start out with a velocity of c in a random direction.

        Parameters
        ----------
        count : int, optional
            How many photons to add, by default 1.
        """
        for i in range(count):
            p = Particle(self)
            p.to_photon()
            p.random_teleport()

    def to_json(self):
        """
        Convert this universe to a dictionary that can then be dumped to a json
        file.

        Returns
        -------
        dict
            Dictionary defining all the parameters and particles of this
            universe.
        """
        out = dict()
        out['particles'] = dict()
        out['size'] = self.size
        for p in self.particles:
            pout = dict()
            pout['type'] = p.type
            pout['m'] = p.m
            pout['q'] = p.q
            pout['x'] = list(p.x)
            pout['v'] = list(p.v)
            pout['last interaction'] = str(p.last_interaction)
            out['particles'][p.name] = pout
        return out


class Particle:
    """
    The `Particle` class contains all the information about a given particle.
    Note that particles cannot exist without being attached to some universe.
    If no universe is provided upon creation of a particle, it will
    automatically generate a new one to put itself into.
    """

    def __init__(self, universe=None):
        """
        Initiate a `Particle` object. New particles are always generated as
        electrons until they are adjusted to have the parameters of some other
        particle.

        Parameters
        ----------
        universe : Universe, optional
            The universe to put the particle into, by default None. If no
            universe is provided, a new one will be created.
        """
        self.type = 'electron'
        self.m = electron_mass
        self.q = -1
        self.x = np.zeros((3))
        self.v = np.zeros((3))
        self.E = np.zeros((3))
        self.B = np.zeros((3))
        self.G = np.zeros((3))
        self.f = np.zeros((3))
        self.interacted = False
        if universe is None:
            universe = Universe()
        self.universe = universe
        self.universe.particles.append(self)
        self.name = 'Particle ' + str(len(self.universe.particles))
        self.last_interaction = None

    def __repr__(self):
        """
        String representation for the `Particle` class.

        Returns
        -------
        str
            A string describing a particle.
        """
        s = self.name
        s += '\n' + str(self.type)
        s += '\nm: ' + str(self.m)
        s += '\nq: ' + str(self.q)
        s += '\nx: ' + str(self.x)
        s += '\nv: ' + str(self.v)
        s += '\nE: ' + str(self.E)
        s += '\nG: ' + str(self.G)
        s += '\nLast Interaction: ' + str(self.last_interaction)
        return s

    def __eq__(self, other):
        """
        Equality operator between two particles. It assumes the particles are
        within the same universe. Each particle within a universe should have
        its own unique (automatically generated) name identifying it. As such,
        this method just checks to see if the two particles have the same name
        or not. Two particles may have exactly the same properties, parameters,
        and locations phase space but would not be declared equal if they do
        not have the same name.

        Parameters
        ----------
        other : Particle
            The particle to compare this particle to.

        Returns
        -------
        bool
            Whether or not this is considered to be equal to the other provided
            particle.
        """
        return self.name == other.name

    def evolve(self, time):
        """
        Evolve this particular particle in time. This is usually called by the
        `evolve` method within a universe and handles interactions for this
        particle. Instead, it assumes that all the relevant fields have been
        generated and applied to this particle previously before time evolution
        is allowed to occur.

        Parameters
        ----------
        time : float
            Time step to evolve for. This is assumed to be in natural units.
        """
        self.f = 0
        fG = self.m * self.G
        fE = self.q * self.E
        f = fG + fE
        self.f = f
        if self.m != 0:
            a = f / self.m
            self.v = vel_add(self.v, a * time)
        self.x += self.v * time
        self.x %= self.universe.size

    def random_teleport(self):
        """
        Move this particle to some random position in the universe.
        """
        x = np.random.random() * self.universe.size
        y = np.random.random() * self.universe.size
        z = np.random.random() * self.universe.size
        self.x = np.array([x, y, z])

    def to_electron(self):
        """
        Convert this particle to an electron.

        Returns
        -------
        Particle
            The converted version of this particle.
        """
        self.type = 'electron'
        momentum = self.m * self.v
        self.m = electron_mass
        self.v = momentum / self.m
        self.q = -1
        return self

    def to_hydrogen(self):
        """
        Convert this particle to a hydrogen atom.

        Returns
        -------
        Particle
            The converted version of this particle.
        """
        self.type = 'hydrogen'
        momentum = self.m * self.v
        self.m = hydrogen_mass
        self.v = momentum / self.m
        self.q = 0
        return self

    def to_photon(self):
        """
        Convert this particle to a photon.

        Returns
        -------
        Particle
            The converted version of this particle.
        """
        self.type = 'photon'
        self.m = 0
        if np.sum(np.abs(self.v)) != 0:
            self.v = c * self.v / np.sqrt(np.sum(self.v ** 2))
        else:
            self.v = c * random_direction()
        self.q = 0
        return self

    def to_proton(self):
        """
        Convert this particle to a proton.

        Returns
        -------
        Particle
            The converted version of this particle.
        """
        self.type = 'proton'
        momentum = self.m * self.v
        self.m = proton_mass
        self.v = momentum / self.m
        self.q = 1
        return self
