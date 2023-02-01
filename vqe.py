# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023 QOTLabs

import numpy as np
from math import *
from scipy.stats import ortho_group
from scipy.optimize import minimize
import logging
import datetime

# Log functions
def log_header(setup, H, logfile):
    logfile.write('# VQE log started ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
    logfile.write('# Hamiltonian = ' + str(H.hamiltonian) + '\n')
    logfile.write('# Exposure time = {:d} ms'.format(setup.exposure_time))
    if(setup.fake):
        logfile.write(', simulations with intensity = {:.1f} Hz, noise = {:.3f} ({:d} channels)\n'
                      .format(setup.intensity, setup.noise, setup.noise_channels))
    else:
        logfile.write('\n')
    logfile.write('# Efficiencies = ' + str(setup.efficiencies) + '\n')
    logfile.write('#')
    for i in range(6):
        logfile.write('{:>7s} '.format('x' + str(i)))
    logfile.write('{:>7s} '.format('Hmean'))
    logfile.write('{:>7s}\n'.format('Hstdev'))
    logfile.flush()

def log_data(logfile, x, res):
    logfile.write(' ')
    for xi in x:
        logfile.write('{:>7.3f} '.format(xi))
    logfile.write('{:>7.3f} '.format(res[0]))
    logfile.write('{:>7.3f}\n'.format(res[1]))
    logfile.flush()

def waveplate(phi, delta):
    """Return waveplate matrix with retardance delta and axis angle phi.
    
    delta = pi for HWP
    delta = pi/2 for QWP
    """
    T = cos(delta/2) + 1j*sin(delta/2)*cos(2*phi)
    R = 1j*sin(delta/2)*sin(2*phi)
    return np.array([[T, R], [-R.conjugate(), T.conjugate()]])

def random_simplex(dim, size):
    """Generate random simplex with the given edge size."""
    O = ortho_group.rvs(dim)
    simplex = size * np.vstack((np.zeros(dim), np.eye(dim))) @ O
    return simplex.tolist()

def apply_krauses(rho0, krauses):
    """Apply Kraus operators to the state rho0.

    krauses - list of Numpy matrices.
    """
    rho = np.zeros_like(rho0)
    for k in krauses:
        rho += k @ rho0 @ k.conj().transpose()
    return rho

def tensordot_krauses(krauses1, krauses2):
    """Calculate tensor dot of two CP maps defined by their Kraus operators.
    """
    E = []
    for k1 in krauses1:
        for k2 in krauses2:
            E.append(np.kron(k1, k2))
    return E

def change_basis(basis, phis, deltas):
    """Return new waveplate angles for given basis.
    
    basis can be either 2-dimensional vector (orthogonal vector is unique),
    or it can be specified by a letter ('h', 'd', 'r' are allowed).
    """
    new_phis = np.copy(phis)
    if type(basis) == str:
        if basis == 'h':
            return new_phis
        elif basis == 'd':
            state = np.array([np.sqrt(0.5), np.sqrt(0.5)])
        elif basis == 'r':
            state = np.array([np.sqrt(0.5), 1j*np.sqrt(0.5)])
        else:
            raise ValueError('Unknown basis type = ' + basis)
    else:
        state = np.copy(basis)
    state = (waveplate(phis[1], deltas[1]) @ waveplate(phis[0], deltas[0])).conj().transpose() @ state
    
    def target_function(new_phis):
        f = np.array([1,0]) @ waveplate(new_phis[1], deltas[1]) @ waveplate(new_phis[0], deltas[0]) @ state
        f = abs(f)**2
        return -f
    
    converged = False
    attempts = 0
    while (not converged) and (attempts < 100):
        #phis0 = np.random.uniform(0, 2*pi, len(phis))
        phis0 = phis
        result = minimize(target_function, phis0)
        converged = (abs(1 + target_function(result.x)) <= 3e-3)
        logging.getLogger(__name__).debug('change_basis() optimizer result: %s', result.fun)
        attempts += 1
    if not converged:
        raise RuntimeError('Cannot change basis after {:d} attempts. Optimizer result:\n{}'.format(attempts, result))
    new_phis = result.x
    return new_phis

def lcwp_noise(eps, delta=2*pi, theta=pi/4):
    """Calculate Kraus operators for liquid crystal waveplate
    with time-dependent retardance.

    eps - noise amplitude in the interval [0..1],
    delta - retardance in radians without noise,
    theta - axis angle in radians.
    """
    s = sin(theta)
    c = cos(theta)
    delta = np.exp(1j*delta)
    V = np.array([[c, -s], [s, c]])
    D1 = sqrt(1-eps/2) * np.diagflat([delta,  1])
    D2 = sqrt(eps/2)   * np.diagflat([delta, -1])
    E = []
    for D in (D1, D2):
        E.append(V @ D @ V.conj().transpose())
    return E

class Setup:
    """Experimental setup.
    
    Waveplate order:
    0 - QWP pump,
    1 - HWP pump,
    2 - QWP 1st qubit,
    3 - HWP 1st qubit,
    4 - QWP 2nd qubit,
    5 - HWP 2nd qubit.
    """
    def find_lbus(self):
        import lbus
        buses = lbus.Lbus.scan()
        motors = lbus.Motors()
        for bus in buses:
            try:
                motors.open(bus, 2)
                motors.close()
                return bus
            except:
                continue
        raise RuntimeError('Cannot find LBUS<=>USB bridges with connected motors')


    def __init__(self, fake = False): 
        """Initialize all hardware"""
        stage_speed = 25 # Degrees/second
        stage_accelaration = 10 # Degrees/second**2
        self.exposure_time = 5000 # Milliseconds

        self.efficiencies = [1, 1, 1, 1]
        self.retardances = [pi/2, pi, pi/2, pi, pi/2, pi] # Waveplace retardances

        # Simulations
        self.fake = fake # Whether to perform simulations
        self.intensity = 1000 # Intensity in Hertz
        self.noise = 0
        self.noise_channels = 1 # 1 - noise in one channel, 2 - noise in both channels
        
        # Private variables
        self._x = None # Current waveplate angles
        self._state = None # State for simulations (before Wollastons)
        
        if not self.fake:
            import lbus
            import thorapt
            self.stage1 = thorapt.PRM1_Z8()
            self.stage1.synchronous = False
            self.stage1.open(0)
            self.stage1.set_vel_params(stage_accelaration, stage_speed)
            self.stage1.move_home(stage_speed)
            
            self.stage2 = thorapt.PRM1_Z8()
            self.stage2.synchronous = False
            self.stage2.open(1)
            self.stage2.set_vel_params(stage_accelaration, stage_speed)
            self.stage2.move_home(stage_speed)
            
            bus = self.find_lbus()
            
            self.correlator = lbus.Correlator()
            self.correlator.open(bus, 1)
            self.set_exposure_time(self.exposure_time)
            
            self.motors1 = lbus.Motors()
            self.motors1.synchronous = False
            self.motors1.open(bus, 2)
            self.motors1.init()
            
            self.motors2 = lbus.Motors()
            self.motors2.synchronous = False
            self.motors2.open(bus, 3)
            self.motors2.init()
            
            self.stage1.wait_for_move_homed()
            self.stage2.wait_for_move_homed()
            self.motors1.wait()
            self.motors2.wait()
        
    def set_exposure_time(self, time):
        """Set exposure time in milliseconds."""
        if not self.fake:
            self.correlator.set_exposure_time(time)
        self.exposure_time = time
        
    def move_directly(self, x):
        """Move waveplates to position specified by a list 'x'.
        
        'x' values are in radians.
        This method tracks previous positions and
        does not perform unnecessary moves.
        """
        should_move = np.not_equal(self._x, x)
        logging.getLogger(__name__).debug('Setup should move motors = %s to positions = %s', should_move, x)
        
        if not self.fake:
            if should_move[0]:
                self.stage1.move_absolute(degrees(x[0]))
            if should_move[1]:
                self.stage2.move_absolute(degrees(x[1]))
            if should_move[2]:
                self.motors1.move(0, degrees(x[2]))
            if should_move[3]:
                self.motors1.move(1, degrees(x[3]))
            if should_move[4]:
                self.motors2.move(0, degrees(x[4]))  
            if should_move[5]:
                self.motors2.move(1, degrees(x[5]))

            if should_move[0]:
                self.stage1.wait_for_move_completed()
            if should_move[1]:
                self.stage2.wait_for_move_completed()
            if should_move[2]:
                self.motors1.wait(0)
            if should_move[3]:
                self.motors1.wait(1)
            if should_move[4]:
                self.motors2.wait(0)
            if should_move[5]:
                self.motors2.wait(1)
        else:
            # Phase between HV and VH components in Sagnac interferometer
            phase = 0.34
            # State after optical isolator
            state = np.array([np.sqrt(0.5), np.sqrt(0.5)])
            # State after pump waveplates
            state = waveplate(x[1], self.retardances[1]) @ waveplate(x[0], self.retardances[0]) @ state
            # State from biphoton source
            state = np.array([0, state[0], state[1]*np.exp(1j*phase), 0])
            # State after fibers

            ########## AM10
            #F1 = np.array([[-0.35969875-0.28782829j, -0.81691947+0.10145661j],
            #    [-0.17727806-0.63515003j,  0.48453841+0.27013763j]])
            #F2 = np.array([[ 0.34477434+0.80391905j,  0.1840585 +0.40692738j],
            #    [ 0.44722742+0.01207619j, -0.8916204 +0.00547886j]])
            ########## AM15
            #F1 = np.array([[-0.1988967 -0.36002104j, -0.82937633+0.15013469j],
            #    [ 0.29940486-0.62012978j,  0.43417327+0.26880498j]])
            #F2 = np.array([[-0.18939352+0.67427504j,  0.15260198+0.38923172j],
            #    [ 0.33831045+0.12757372j, -0.86878479+0.0924475j ]])
            ########## AM50
            #F1 = np.array([[-0.30043639+0.04905967j,  0.15051634-0.37097088j],
            #    [-0.43252532+0.11909014j,  0.13425901+0.01458326j]])
            #F2 = np.array([[ 0.28777641+0.50457101j, -0.08155068-0.34105686j],
            #    [-0.6524402 +0.09200715j, -0.56266806-0.04711743j]])
            #state = np.kron(F1, F2) @ state
            
            # Transform to density matrix
            state = np.tensordot(state, state.conjugate(), axes=0)
            # State before Wollastons
            U1 = waveplate(x[3], self.retardances[3]) @ waveplate(x[2], self.retardances[2])
            U2 = waveplate(x[5], self.retardances[5]) @ waveplate(x[4], self.retardances[4])
            U12 = np.kron(U1, U2)
            state = U12 @ state @ U12.conj().transpose()
            # Apply noise if required
            if(self.noise != 0):
                k1 = lcwp_noise(self.noise, delta=2*pi, theta=pi/4)
                if self.noise_channels == 1:
                    k2 = [np.eye(2)] # 2nd qubit without noise
                elif self.noise_channels == 2:
                    k2 = k1 # Same noise for the 2nd qubit
                else:
                    raise RuntimeError('Wrong noise_channels = {:d} option value!'.format(self.noise_channels))
                state = apply_krauses(state, tensordot_krauses(k1, k2))

            # Save final state
            self._state = state
        
        logging.getLogger(__name__).debug('Setup move finished')            
        self._x = np.copy(x)
        
    def move(self, x, basis):
        """Move waveplates to position 'x' taking into account specified basis."""
        xnew = np.copy(x)
        xnew[2:4] = change_basis(basis[0], x[2:4], self.retardances[2:4])
        xnew[4:6] = change_basis(basis[1], x[4:6], self.retardances[4:6])
        logging.getLogger(__name__).debug('Setup found new angles = %s for basis = %s (old angles = %s)', xnew, basis, x)
        self.move_directly(xnew)
        
    def measure(self):
        """Return measured counts taking into account detection efficiencies."""
        if not self.fake:
            self.correlator.start()
            r = self.correlator.get_results()
            raw_counts = np.array(r.coins)
        else:
            p = np.abs(self._state.diagonal())
            logging.getLogger(__name__).debug('Calculated probabilities = %s', p)
            mean = self.intensity * self.exposure_time/1000 * p / self.efficiencies
            raw_counts = np.random.poisson(mean)

        counts = raw_counts * self.efficiencies
        logging.getLogger(__name__).info('Measured raw counts = %s, effective counts = %s', raw_counts, counts)
        return counts

class MeanValue:
    """Mean Hamiltonian value evaluator."""
    def __init__(self, setup, hamiltonian):
        self.setup = setup
        self.hamiltonian = hamiltonian
        self.cache = {}
        self.hits = 0 # Cache hits
        self.miss = 0 # Cache misses
        
    def measure(self, x):
        """Measure mean value.
        
        Return a tuple (mean, stdev),
        where stdev is estimated assuming Poissonian statistics
        """
        mean = 0
        stdev = 0
        for basis, coeffs in self.hamiltonian.items():
            self.setup.move(x, basis)
            counts = self.setup.measure()
            m = counts @ coeffs / sum(counts)
            s = ((coeffs - m)**2) @ counts / (sum(counts)**2)
            mean += m
            stdev += s
        stdev = np.sqrt(stdev)
        return (mean, stdev)

    def __call__(self, x):
        """Measure mean value with caching (preferred).
        
        See measure() method for details.
        """
        key = tuple(x)
        if key in self.cache:
            msg = 'cached'
            self.hits += 1
            result = self.cache[key]
        else:
            msg = 'measured'
            self.miss += 1
            result = self.measure(x)
            self.cache[key] = result
        logging.getLogger(__name__).info('MeanValue(%s) = %s (%s)', x, result, msg)
        return result
    
    @property
    def cache_stats(self):
        return 'hits = %s, miss = %s, size = %s' % (self.hits, self.miss, len(self.cache))
