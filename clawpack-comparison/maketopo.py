"""
Module to create topo and qinit data files for this example.
"""

from __future__ import absolute_import
from clawpack.geoclaw.topotools import Topography
from osgeo import gdal,ogr
gdal.UseExceptions()
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import scipy.interpolate
from scipy.signal import convolve2d
#from pyclaw.data import Data
#probdata = Data('setprob.data')

a = 1.
sigma = 0.5
h0 = 0.1
grav = 9.81
omega = sqrt(2.*grav*h0) / a

driver = gdal.GetDriverByName('GTiff')
GDT_dtype = gdal.GDT_Float32
raster = gdal.Open('crescent_city_harbor_rotated.tif')
conv_map = raster.ReadAsArray()
conv_map = convolve2d(conv_map,np.array([[1,1,1],[1,1,1],[1,1,1]])/9.0,mode='same',boundary='symm')
resolution = 8
geoinfo = raster.GetGeoTransform()    
base_lx = geoinfo[1] * conv_map.shape[0]
base_ly = geoinfo[5] * conv_map.shape[1]
ny = int(base_lx / resolution) + 1
nx = int(base_ly / resolution) + 1
print(ny)
print(nx)
base_x = np.linspace(0, base_lx,conv_map.shape[0])
base_y = np.linspace(0, base_ly,conv_map.shape[1])

dem_function = scipy.interpolate.interp2d(base_y,base_x,conv_map,kind='cubic')


def maketopo(makeplots=False):
    """
    Output topography file for the entire domain
    """

    outfile= "crescent_city_harbor_rotated_upres.topo2"

    topography = Topography(topo_func=topo)
    topography.x = np.linspace(0, base_ly, nx)
    topography.y = np.linspace(0, base_lx, ny)
    topography.write(outfile, topo_type=2, Z_format="%22.15e")
    if makeplots:
        from matplotlib import pyplot as plt
        topography.plot()
        fname = 'crescent_city_output.png'
        plt.savefig(fname)

def makeqinit(qinitFunc, outfile,makeplots=False):
    topography = Topography(topo_func=qinitFunc)
    topography.x = np.linspace(0, base_ly, nx)
    topography.y = np.linspace(0, base_lx, ny)
    topography.write(outfile, topo_type=1)
    if makeplots:
        from matplotlib import pyplot as plt
        topography.plot()
        fname = 'crescent_city_ic_output.png'
        plt.savefig(fname)

def topo(x,y):
    to = dem_function(x[1,:],y[:,1])
    return to

def soliton_scenario(x,y):
    A = 10
    angle = -15 * 3.141592653 / 180.0
    lamb = 2000
  #	A = 6
    #	-	Case 3
  #	A = 30
    nx = 1992
    ny = 901
    L = 15928
    # N-wave parameters
    x_proj = x*(np.cos(angle))-y*np.sin(angle)
    a1 = A
    a2 = A/3
    x1 = L - (0.5151125 * lamb + 6000)

    x2 = L - (0.2048 * lamb + 6000)		#	Leading Depression Nwave
  #	x2 = 1000 + 0.825425 * L 	#	Leading Elevation Nwave

    k1 = 28.416 / lamb ** 2
    k2 = 256 / lamb ** 2
    z = 2 * (a1 * np.exp( -k1 * np.square(x_proj - x1) ) - a2 * np.exp( -k2 * np.square(x_proj - x2)))
    # z = np.ones_like(x)*5.0
    return z

if __name__=='__main__':
    maketopo(True)
    makeqinit(soliton_scenario, 'crescent_city_ic.xyz',True)