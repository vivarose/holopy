# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
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
"""
Routines for common calculations and transformations of groups of spheres.

.. moduleauthor:: Rebecca W. Perry <rperry@seas.harvard.edu>

"""

import numpy as np
from numpy import sqrt
import holopy as hp
from scatterpy.scatterer import Sphere, SphereCluster
from matplotlib import pyplot

#calculate the distances between each sphere in a cluster and each
#of the others
def distances(cluster, gaponly=False):
    """
    Parameters
    ----------
    cluster: :class:`scatterpy.scatterer.Scatterer`
        A sphere cluster to determine the interparticle distances of.
    gaponly: bool
        Whether to calculate the distances between particle centers
        or between particle surfaces (gap distances).
        
    Notes
    -----
    The returned array of distances includes redundant information.
    The identical distances between sphere 1 and sphere 2 and between sphere 2
    and sphere 1 are both in the returned array. Calculating and returning
    the full array makes it easy for the user to access all the interparticle 
    distances starting from any sphere of interest.
    
    """
    num = len(cluster.centers)
    dist = np.zeros([num,num])
    for i in np.arange(0,num):
        for j in np.arange(0,num):
            dist[i,j] = hp.process.cartesian_distance(cluster.centers[i,:],cluster.centers[j,:])
            #modification to change center to center distances to gap distances if asked for
            if gaponly==True and i!=j:
                dist[i,j] = dist[i,j]-cluster.r[i]-cluster.r[j]
    return dist

#calculate the angles between one particle and every pair
#of other particles
def angles(cluster, degrees=True):
    """
    Parameters
    ----------
    cluster: :class:`scatterpy.scatterer.Scatterer`
        A sphere cluster to determine the interparticle distances of.
    degrees: bool
        Whether to return angles in degrees (True) or in radians (False).
        
    Notes
    -----
    Angle abc is the acute angle formed by edges conecting points ab and bc.
    If a, b, and c are locations of particles (vertices),
    the returned 3D array has non-zero values for angles abc, zeros
    for angles aba, and NAN's for "angles" aab.
    
    """
    num = len(cluster.centers)
    ang = np.zeros([num,num,num])
    dist = distances(cluster)
    for i in np.arange(0,num):
        for j in np.arange(0,num): #this particle is the center of the angle
            for k in np.arange(0,num):
                Adjacent1=dist[i,j]
                Adjacent2=dist[j,k]
                Opposite=dist[i,k]
                #use the law of cosines to determine the angles from the distances
                ang[i,j,k] = np.arccos((Adjacent1**2+Adjacent2**2-Opposite**2)/(2*Adjacent1*Adjacent2))
    if degrees==True:
        ang=ang/np.pi*180.0
    return ang #ang[a,b,c] is the acute angle abc as used in geometry (be in the middle)

def make_tricluster(index,radius,gap):
    """
    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
        
    Notes
    -----
    Returns a sphere cluster of three particles forming an equilateral triangle
    centered on (0,0,0).
    
    """
    #currently restricted to all identical spheres
    xs = [1/sqrt(3)*(radius+gap/2.0),1/sqrt(3)*(radius+gap/2.0),-2/sqrt(3)*(radius+gap/2.0)]
    ys = [-radius-gap/2.0,radius+gap/2.0,0]
    zs = np.zeros(3)
    triangle = SphereCluster([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2]))])
    return triangle

def make_sqcluster(index,radius,gap):
    """
    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
        
    Notes
    -----
    Returns a sphere cluster of four particles forming a square
    centered on (0,0,0).
    """
    #currently restricted to all identical spheres
    xs = [-radius-gap/2.0,-radius-gap/2.0,radius+gap/2.0,radius+gap/2.0]
    ys = [-radius-gap/2.0, radius+gap/2.0,-radius-gap/2.0,radius+gap/2.0]
    zs = [a,a,a,b]
    square = SphereCluster([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3]))])
    return square

def make_tetracluster(index,radius,gap):
    """
    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
        
    Notes
    -----
    Returns a sphere cluster of four particles forming a tetrahedron
    centered on (0,0,0).
    """
    #currently restricted to all identical spheres
    xs = [1/sqrt(3)*(radius+gap/2.0),1/sqrt(3)*(radius+gap/2.0),-2/sqrt(3)*(radius+gap/2.0),0]
    ys = [-radius-gap/2.0,radius+gap/2.0,0,0]
    zs = [-(1/4.0)*sqrt(2/3.0)*(2*radius+gap),-(1/4.0)*sqrt(2/3.0)*(2*radius+gap),-(1/4.0)*sqrt(2/3.0)*(2*radius+gap),(3/4.0)*sqrt(2/3.0)*(2*radius+gap)]
    tetra = SphereCluster([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3]))])
    return tetra

def make_tribipyrcluster(index,radius,gap):
    """
    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
        
    Notes
    -----
    Returns a sphere cluster of five particles forming a triagonal bipyramid
    centered on (0,0,0).
    """
    #currently restricted to all identical spheres
    xs = [1/sqrt(3)*(radius+gap/2.0),1/sqrt(3)*(radius+gap/2.0),-2/sqrt(3)*(radius+gap/2.0),0,0]
    ys = [-radius-gap/2.0,radius+gap/2.0,0,0,0]
    zs = [0,0,0,sqrt(2/3.0)*(2*radius+gap),-sqrt(2/3.0)*(2*radius+gap)]
    triangularbipyramid = SphereCluster([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3])),
        Sphere(n=index, r = radius, center = (xs[4], ys[4], zs[4]))])
    return triangularbipyramid

def make_octacluster(index,radius,gap):
    """
    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
        
    Notes
    -----
    Returns a sphere cluster of six particles forming an octahedron
    centered on (0,0,0).
    """
    #currently restricted to all identical spheres
    xs = [-radius-gap/2.0,-radius-gap/2.0,radius+gap/2.0,radius+gap/2.0,0,0]
    ys = [-radius-gap/2.0, radius+gap/2.0,-radius-gap/2.0,radius+gap/2.0,0,0]
    zs = [0,0,0,0,1/sqrt(2)*(2*radius+gap),-1/sqrt(2)*(2*radius+gap)]
    octahedron = SphereCluster([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3])),
        Sphere(n=index, r = radius, center = (xs[4], ys[4], zs[4])),
        Sphere(n=index, r = radius, center = (xs[5], ys[5], zs[5]))])
    return octahedron

def make_cubecluster(index,radius,gap):
    """
    Parameters
    ----------
    index:
        Index of refraction of particles.
    radius:
        Radius if particles.
    gap:
        Space to add between the particles.
        
    Notes
    -----
    Returns a sphere cluster of eight particles forming a cube
    centered on (0,0,0).
    """
    #currently restricted to all identical spheres
    xs = [-radius-gap/2.0,-radius-gap/2.0,radius+gap/2.0,radius+gap/2.0,
            -radius-gap/2.0,-radius-gap/2.0,radius+gap/2.0,radius+gap/2.0]
    ys = [-radius-gap/2.0, radius+gap/2.0,-radius-gap/2.0,radius+gap/2.0,
            -radius-gap/2.0, radius+gap/2.0,-radius-gap/2.0,radius+gap/2.0]
    zs = [-radius-gap/2.0,-radius-gap/2.0,-radius-gap/2.0,-radius-gap/2.0,
            radius+gap/2.0,radius+gap/2.0,radius+gap/2.0,radius+gap/2.0]
    cube = SphereCluster([
        Sphere(n=index, r = radius, center = (xs[0], ys[0], zs[0])),
        Sphere(n=index, r = radius, center = (xs[1], ys[1], zs[1])),
        Sphere(n=index, r = radius, center = (xs[2], ys[2], zs[2])),
        Sphere(n=index, r = radius, center = (xs[3], ys[3], zs[3])),
        Sphere(n=index, r = radius, center = (xs[4], ys[4], zs[4])),
        Sphere(n=index, r = radius, center = (xs[5], ys[5], zs[5])),
        Sphere(n=index, r = radius, center = (xs[6], ys[6], zs[6])),
        Sphere(n=index, r = radius, center = (xs[7], ys[7], zs[7]))])
    return cube

def viewcluster(cluster):
    #this is not elegant, but lets you look at the cluster from three angles
    #to check if it is the cluster you wanted
    #warning: the particles are not shown to scale!!!!!! (markersize is in points)
    dist = distances(cluster)
    pyplot.figure(figsize=[14,4])
    pyplot.subplot(1,3,1)
    l=pyplot.plot(cluster.centers[:,0],cluster.centers[:,1],'ro')
    pyplot.setp(l, 'markersize', 60)
    pyplot.xlim(-dist.max(),dist.max())
    pyplot.ylim(-dist.max(),dist.max())
    pyplot.subplot(1,3,2)
    l=pyplot.plot(cluster.centers[:,0],cluster.centers[:,2],'ro')
    pyplot.setp(l, 'markersize', 60)
    pyplot.xlim(min(pyplot.xlim()[0],pyplot.ylim()[0]),max(pyplot.xlim()[1],pyplot.ylim()[1]))
    pyplot.ylim(min(pyplot.xlim()[0],pyplot.ylim()[0]),max(pyplot.xlim()[1],pyplot.ylim()[1]))
    pyplot.xlim(-dist.max(),dist.max())
    pyplot.ylim(-dist.max(),dist.max())
    pyplot.subplot(1,3,3)
    l=pyplot.plot(cluster.centers[:,1],cluster.centers[:,2],'ro')
    pyplot.setp(l, 'markersize', 60)
    pyplot.xlim(min(pyplot.xlim()[0],pyplot.ylim()[0]),max(pyplot.xlim()[1],pyplot.ylim()[1]))
    pyplot.ylim(min(pyplot.xlim()[0],pyplot.ylim()[0]),max(pyplot.xlim()[1],pyplot.ylim()[1]))
    pyplot.xlim(-dist.max(),dist.max())
    pyplot.ylim(-dist.max(),dist.max())