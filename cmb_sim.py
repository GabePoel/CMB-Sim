#!/usr/bin/python3

import numpy as np
import multiprocessing as mp
from scipy.special import zeta

workers = 3
k = 1
c = 1
G = 0.0001
vcut = 0.9 * c
# G = 0
hbar = 1
electric_constant = 1
electron_mass = 1
bohr_radius = 1
alpha = 1 / 137
# alpha = e^2 / (4 * np.pi)
hydrogen_mass = 1837.5 * electron_mass
proton_mass = 1836.15267 * electron_mass
photon_mass = hydrogen_mass - proton_mass - electron_mass
photon_energy = np.abs(proton_mass + electron_mass - hydrogen_mass) * c ** 2
interaction_rate = 0.5
interaction_radius = 0.1


def vel_add(v1, v2):
    return (v1 + v2) / (1 + (v1 * v2 / c ** 2))


def dist(p1, p2):
    """
    Find cartesian distance between two particles modulo the size of the
    universe.

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


def check_types(particles, types):
    included = [p.type for p in particles]
    return all([t in included for t in types])


def sort_types(particles, types):
    sorted_particles = []
    for t in types:
        for p in particles:
            if p.type == t and p not in sorted_particles:
                sorted_particles.append(p)
    for p in particles:
        if p not in sorted_particles:
            sorted_particles.append(p)
    return sorted_particles


def thomson_radius(particle):
    if particle.type == 'photon':
        return 0
    # Radius increaser
    elif particle.type == 'electron':
        return 0.05
    m = particle.m
    return 8 * np.pi / 3 * (alpha * hbar * c / (m * c ** 2)) ** 2


def smoother(x):
    return -1 / (x + 1) + 1


def truncate_velocity(v):
    mag = np.sqrt(np.sum(v) ** 2)
    if mag > vcut:
        new_mag = (1 - vcut) * smoother((mag - vcut) / (1 - vcut)) + vcut
        # new_mag = vcut
        v = new_mag * v / mag
    return v


def random_direction():
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
    Evolve particle in time due to the forces on it.
    """
    def __init__(self, time):
        self.time = time

    def __call__(self, particle):
        particle.evolve(self.time)
        particle.G = np.zeros((3))
        particle.E = np.zeros((3))


class Graviter:
    """
    Determine gravitational field at site of particle.
    """
    def __init__(self, var):
        pass

    def __call__(self, particles):
        p1, p2 = particles[0], particles[1]
        if p1 is not p2:
            G0 = -G * p2.m * (p1.x - p2.x) / dist(p1, p2) ** 3
            p1.G += G0


class Electricer:
    """
    Determine electric field at site of particle.
    """
    def __init__(self, var):
        pass

    def __call__(self, particles):
        p1, p2 = particles[0], particles[1]
        if p1 is not p2:
            E0 = 1 / (4 * np.pi * electric_constant) * \
                p2.q * (p1.x - p2.x) / dist(p1, p2) ** 3
            p1.E += E0


class Interacter:
    def __init__(self, var):
        pass

    def __call__(self, particles):
        p1, p2 = particles[0], particles[1]
        if p1 is not p2:
            p1.universe.thomson(p1, p2)
            if np.random.random() <= interaction_rate and dist(p1, p2) <= interaction_radius:
                p1.universe.ionization(p1, p2)
                p1.universe.recombination(p1, p2)


class Measurer:
    def __init__(self, var):
        pass

    def __call__(self, particles):
        p1, p2 = particles[0], particles[1]
        return (p1, p2, dist(p1, p2))


class Universe:
    def __init__(
            self,
            size=1,
            workers=1,
            mpG=True,
            mpE=True,
            mpEvolve=True,
            mpInteract=True,
            mpMeasure=True):
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

    def __repr__(self):
        s = 'Size: ' + str(self.size)
        s += '\nT: ' + str(self.T)
        for particle in self.particles:
            s += '\n\n' + str(particle)
        return s

    def interface_func(self, f, x):
        pool = mp.Pool(self.workers)
        return list(pool.map(f, x))

    def evolve(self, time):
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
        self.size = self.next_size

    def stage_interactions(self):
        if self.mpMeasure:
            p_table = []
            for p1 in self.particles:
                for p2 in self.particles:
                    p_table.append((p1, p2))
            measurer = MultiprocessHelper(1, Measurer).make_caller()
            ps = self.interface_func(measurer, p_table)
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
            self.interface_func(electricer, p_table)
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
        Number of photons.
        """
        return np.sum([1 for p in self.particles if p.type == 'photon'])

    def electron_count(self):
        """
        Number of photons.
        """
        return np.sum([1 for p in self.particles if p.type == 'electron'])

    def proton_count(self):
        """
        Number of photons.
        """
        return np.sum([1 for p in self.particles if p.type == 'proton'])

    def hydrogen_count(self):
        """
        Number of photons.
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
        for i in range(count):
            p = Particle(self)
            p.to_electron()
            p.random_teleport()

    def add_protons(self, count=1):
        for i in range(count):
            p = Particle(self)
            p.to_proton()
            p.random_teleport()

    def add_hydrogens(self, count=1):
        for i in range(count):
            p = Particle(self)
            p.to_hydrogen()
            p.random_teleport()

    def add_photons(self, count=1):
        for i in range(count):
            p = Particle(self)
            p.to_photon()
            p.random_teleport()


class Particle:
    def __init__(self, universe=None):
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

    def evolve(self, time):
        self.f = 0
        fG = self.m * self.G
        fE = self.q * self.E
        f = fG + fE
        self.f = f
        if self.m != 0:
            a = f / self.m
            self.v = vel_add(self.v, a * time)
            # self.v = truncate_velocity(self.v)
        self.x += self.v * time
        self.x %= self.universe.size

    def random_teleport(self):
        x = np.random.random() * self.universe.size
        y = np.random.random() * self.universe.size
        z = np.random.random() * self.universe.size
        self.x = np.array([x, y, z])

    def to_electron(self):
        self.type = 'electron'
        momentum = self.m * self.v
        self.m = electron_mass
        self.v = momentum / self.m
        self.q = -1
        return self

    def to_hydrogen(self):
        self.type = 'hydrogen'
        momentum = self.m * self.v
        self.m = hydrogen_mass
        self.v = momentum / self.m
        self.q = 0
        return self

    def to_photon(self):
        self.type = 'photon'
        self.m = 0
        if np.sum(np.abs(self.v)) != 0:
            self.v = c * self.v / np.sqrt(np.sum(self.v ** 2))
        else:
            self.v = c * random_direction()
        self.q = 0
        return self

    def to_proton(self):
        self.type = 'proton'
        momentum = self.m * self.v
        self.m = proton_mass
        self.v = momentum / self.m
        self.q = 1
        return self
