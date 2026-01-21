import numpy as np
import math as ma
import json
import os

class ShkuratovModel3p(object):
    """ This is a python class defining the Shkuratov physical model. 
    
    This class is composed of 5 mandatory functions:
        - F: the functional model F describing the physical model. F takes photometries as arguments and return reflectances
        - getDimensionY: returns the dimension of Y (reflectances)
        - getDimensionX: return de dimension of X (photometries)
        - toPhysic: converts the X data from mathematical framework (0<X<1) to physical framework
        - fromPhysic: converts the X data from physical framework to mathematical framework (0<X<1)

    Note that some class constants, other functions and class constructors can be declared.
    """
    
    #################################################################################################
    ##                          CLASS CONSTANTS (OPTIONAL)                                         ##
    #################################################################################################

    DEGREE_180 = 180
    L_dimension = 3
    scalingCoeffs = [1.0,1.5,1.3]
    offset = [0,0,0.2]

    # geometry
    ALPHA = 0
    BETA = 1
    GAMMA = 2
    INC = 0 # sza
    EME = 1 # vza
    PHI = 2 # phi

    # photometry
    AN = 0
    MU_1 = 1
    NU = 2
 
    #################################################################################################
    ##                          CORE FUNCTIONS (MANDATORY)                                         ##
    #################################################################################################

    def F(self, photometry):
        photometry = self.toPhysic(photometry)

        f = np.exp(- photometry[self.MU_1] * self.configuredGeometries[self.ALPHA]) 
        d = np.cos(self.configuredGeometries[self.ALPHA] / 2.0) * np.cos(np.pi * (self.configuredGeometries[self.GAMMA] - self.configuredGeometries[self.ALPHA] / 2.0) / (np.pi - self.configuredGeometries[self.ALPHA])) / np.cos(self.configuredGeometries[self.GAMMA])
        o=d*np.cos(self.configuredGeometries[self.BETA])**(photometry[self.NU] * self.configuredGeometries[self.ALPHA] * (np.pi - self.configuredGeometries[self.ALPHA]))
        reflectances = photometry[self.AN] * o.T * f.T / self.cos_i.T

        return reflectances


    def getDimensionY(self):
        return self.D_dimension

    def getDimensionX(self):
        return self.L_dimension

    def toPhysic(self, x):
        xp=np.copy(x)
        for l in range(x.shape[0]):
            xp[l] = xp[l] * self.scalingCoeffs[l] + self.offset[l]
        return xp

    def fromPhysic(self, xp):
        xp=np.copy(x)
        for l in range(x.shape[0]):
            xp[l] = (xp[l] - self.offset[l]) / self.scalingCoeffs[l]
        return xp


    #################################################################################################
    ##                          OTHER FUNCTIONS (OPTIONAL)                                         ##
    #################################################################################################

    def __init__(self):
        # geometries data
        print("Entering __init__")
        geom_tmp = []
        geom_tmp.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,40,40,40,40,40,40,40,40,40,40,40,40,40,40,60,60,60,60,60,60,60,60,60,60,60,60,60,60,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20])
        geom_tmp.append([70,60,50,40,30,20,10,10,20,30,40,50,60,70,70,60,50,40,30,20,10,0,10,30,40,50,60,70,70,60,50,30,20,10,0,10,20,30,40,50,60,70,70,60,50,40,30,20,10,0,10,20,30,40,50,70,70,60,50,40,30,20,10,0,10,20,30,40,50,60,70])
        geom_tmp.append([0,0,0,0,0,0,0,180,180,180,180,180,180,180,180,180,180,180,180,180,180,0,0,0,0,0,0,0,0,0,0,0,0,0,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,0,0,0,0,0,0,30,30,30,30,30,30,30,150,150,150,150,150,150,150,150])
        geometries=np.array(geom_tmp)
        self.D_dimension=geometries.shape[1]
        self.configuredGeometries=self.setupGeometries(geometries)
        self.cos_i=np.cos(geometries[self.INC]*np.pi / self.DEGREE_180)
        print("Leaving __init__")

    def degToGrad(self, degree):
        return degree * np.pi / self.DEGREE_180


    def setupGeometries(self, geometries):
        configuredGeometries = np.zeros(geometries.shape)
        geomsGrad = np.array(geometries)
        geomsGrad = self.degToGrad(geomsGrad)

        #compute Alpha
        configuredGeometries[self.ALPHA] = np.arccos(np.cos(geomsGrad[self.INC]) * np.cos(geomsGrad[self.EME]) + np.sin(geomsGrad[self.INC]) * np.sin(geomsGrad[self.EME]) * np.cos(geomsGrad[self.PHI]))

        #compute Beta
        sin_i_e_2 = pow(np.sin(geomsGrad[self.INC] + geomsGrad[self.EME]),2)
        cos_phiDiv2_2 = pow(np.cos(geomsGrad[self.PHI]/2.0),2)
        sin_2_i = np.sin(geomsGrad[self.INC] * 2)
        sin_2_e = np.sin(geomsGrad[self.EME] * 2)
        cos_beta = np.sqrt(
                (sin_i_e_2 - cos_phiDiv2_2 * sin_2_i * sin_2_e) /
                (sin_i_e_2 - cos_phiDiv2_2 * sin_2_i * sin_2_e + pow(np.sin(geomsGrad[self.EME]),2) * pow(np.sin(geomsGrad[self.INC]),2) * pow(np.sin(geomsGrad[self.PHI]),2)))
        configuredGeometries[self.BETA] = np.arccos(cos_beta)

        #compute Gamma
        #tmp_angle=np.arccos(np.cos(geomsGrad[self.EME]) / cos_beta)
        #configuredGeometries[self.GAMMA] = np.where(geomsGrad[self.PHI]<ma.pi/2.,-tmp_angle,tmp_angle)
        configuredGeometries[self.GAMMA] = np.arctan((np.cos(geomsGrad[self.INC])/np.cos(geomsGrad[self.EME])-np.cos(configuredGeometries[self.ALPHA]))/np.sin(configuredGeometries[self.ALPHA]))

        return configuredGeometries
