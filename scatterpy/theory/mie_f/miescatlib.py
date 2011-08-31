# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.

'''
MieScatLib.py

Library of code to do Mie scattering calculations

Caution: these use SciPy's special function libraries; the code is
hidden in Fortran.  It is not clear what numerical methods are used
(upwards vs. downwards recursion).  Numerical stability not guaranteed
for large nstop, so do not use for calculating very large size
parameters.

05-18-08: Started by JF
05-26-08: Bug discovered in LogDerPsi (it does not correctly deal
w/complex argument). Fixed by writing LogDerPsi2 (calculates
logarithmic derivative by downward recursion, as in BHMIE).
05-27-08: Added calculation of backscattering cross section
07-07-08: Generalized RicBesHank to handle Riccati-Bessel functions of
complex arguments
'''

import scipy
import mieangfuncs # use Fortran angular functions to avoid duplication

from scipy import sin, cos, array
from scipy.special import lpn, riccati_jn, riccati_yn, sph_jn, sph_yn


def RicBesHank(x, nstop): 
    # modification: generalize to allow for complex x
    if scipy.imag(x) == 0:
        psin = riccati_jn(nstop, x)
        # construct riccati hankel function of 1st kind by linear
        # combination of RB's based on j_n and y_n 
        # scipy sign on y_n consistent with B/H
        xin = riccati_jn(nstop, x)[0] + 1j*riccati_yn(nstop, x)[0] 
        rbh = array([psin[0], xin])
    else:
        rbjns = x*sph_jn(nstop, x)[0]
        rbyns = x*sph_yn(nstop, x)[0]
        rbh = array([rbjns, (rbjns+1j*rbyns)])
    return rbh


def LogDerPsi2(z, nmx, nstop):
    # function LogDerPsi gives incorrect values for complex argument
    # calculate instead using downward recursion as in BHMIE
    # we could use the method of Lentz to more accurately calculate
    # the highest order of D_n(z), but it's not clear how much
    # would be gained  
    dn = scipy.zeros(nmx+1, dtype = 'complex128')
    # initialize w/zeros
    for i in scipy.arange(nmx-1, -1, -1):
        # 1's must be floats to avoid division problems
        dn[i] = (i+1.)/z - 1.0/(dn[i+1.] + (i+1.)/z)
    #for i in scipy.arange(1, nmx+1):
    #	dn[nmx-i] = (nmx-i+1.)/z - 1./(dn[nmx-i+1.] + (nmx-i+1.)/z )
    return dn[0:nstop+1]


def scatcoeffs(x, m, nstop): # see B/H eqn 4.88
    # implement criterion used by BHMIE plus a couple more orders to
    # be safe 
    nmx = array([nstop, scipy.round_(scipy.absolute(m*x))]).max() + 20
    Dnmx = LogDerPsi2(m*x, nmx, nstop) # corrected version w/down recurrence 
    n = scipy.arange(nstop+1)
    psiandxi = RicBesHank(x, nstop)
    psi = psiandxi[0]
    xi = psiandxi[1]
    psishift = scipy.concatenate((scipy.zeros(1), psi))[0:nstop+1]
    xishift = scipy.concatenate((scipy.zeros(1), xi))[0:nstop+1]
    an = ( (Dnmx/m + n/x)*psi - psishift ) / ( (Dnmx/m + n/x)*xi - xishift )
    bn = ( (Dnmx*m + n/x)*psi - psishift ) / ( (Dnmx*m + n/x)*xi - xishift )
    return scipy.array([an[1:nstop+1], bn[1:nstop+1]]) # output begins at n=1


def nstop(x): 
    #takes size parameter, outputs order to compute to according to
    # Wiscombe, Applied Optics 19, 1505 (1980).
    # 7/7/08: generalize to apply same criterion when x is complex
    return scipy.round_(scipy.absolute(x+4.05*x**(1./3.)+2))


def asymmetry_parameter(an, bn):
    '''
    Inputs: an, bn coefficient arrays from Mie solution
    
    See discussion on Bohren & Huffman p. 120.
    The output of this function omits the prefactor of 4/(x^2 Q_sca).
    '''
    nmax = an.shape[0]
    n = scipy.arange(nmax) + 1
    astarshift = scipy.concatenate((an.conj()[1:], scipy.zeros(1)))
    bstarshift = scipy.concatenate((bn.conj()[1:], scipy.zeros(1)))
    gterms = n * (n+2.) / (n+1.) * (an*astarshift + bn*bstarshift).real + (
        2.*n + 1.) / (n*(n+1.)) * (an*bn.conj()).real

    return gterms.sum()


def cross_sections(an, bn): 
    '''
    Calculates scattering, extension, and radar backscattering cross sections
    given arrays of Mie scattering coefficients an and bn.

    See Bohren & Huffman eqns. 4.61 and 4.62.

    The output omits a scaling prefactor of 2 * pi / k^2.
    '''
    nmax = an.shape[0] # determine number of terms in series
    alts = 2*(scipy.arange(nmax)%2) - 1 # mod 2 division gets alternating signs
    prefactor = 2*(scipy.arange(nmax)+1)+1
    cscacoeffs = an.conj()*an + bn.conj()*bn
    csca = (prefactor*cscacoeffs).sum()
    cextcoeffs = (an+bn).real	
    cext = (prefactor*cextcoeffs).sum()
    cbackcoeffs = an - bn
    cback = scipy.absolute((prefactor*alts*cbackcoeffs).sum())**2
    return array([csca, cext, cback], dtype = 'float64') # contents are real


def ascatmatrix_mie(theta, a_l, b_l):
    '''
    Calculate nonzero (diagonal) elements of the amplitude scattering matrix
    for a sphere or coated sphere.

    Input: scattering angle and scattering coefficients a_l and b_l

    From the amplitude scattering matrix one can calculate fields for
    arbitrary polarization.
    '''
    nstop = a_l.size
   
    n = scipy.arange(nstop)+1.
    prefactor = (2*n+1.)/(n*(n+1.))

    angfuncs = mieangfuncs.pisandtaus(nstop, theta)
    pis = angfuncs[0]
    taus = angfuncs[1]
    
    S1 = (prefactor*(a_l*pis + b_l*taus)).sum()
    S2 = (prefactor*(a_l*taus + b_l[1]*pis)).sum()
    return array([S2, S1])