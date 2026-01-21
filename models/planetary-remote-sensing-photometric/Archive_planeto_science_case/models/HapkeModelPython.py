import numpy as np
import math as ma
import json
import os

class HapkeModel6p(object):
    """ This is a python class defining the Hapke physical model. 
    
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
    L_dimension = 6
    scalingCoeffs = [1.0,30.,1.0,1.0,1.0,1.0]
    offset = [0.,0.,0.,0.,0.,0.]

    # geometry
    INC = 0 # sza
    EME = 1 # vza
    PHI = 2 # phi

    # photometry
    W = 0
    R = 1
    BB = 2
    CC = 3
    B0 = 4
    HH = 5

    def F(self, photometry):
        photometry = self.toPhysic(photometry)

        # This function calculates the reflectance value according to the Hapke's model
        # Inputs:
        # SZA : solar zenith angle in degrees
        # VZA : view zenith angle in degrees
        # DPHI : azimuth in degrees
        # W : single scattering albedo
        # R : macroscopic roughness
        # BB : asymmetry of the phase function
        # CC : fraction of the backward scattering
        # HH : angular width of the opposition effect
        # B0 : amplitude of the opposition effect
        #
        # Output:
        # reflectances : bidirectional reflectance factor

        W=photometry[self.W]
        R=photometry[self.R]
        BB=photometry[self.BB]
        CC=photometry[self.CC]
        B0=photometry[self.B0]
        HH=photometry[self.HH]
        
        MUP = np.cos(np.radians(self.geometries[self.INC,:]))
        MU = np.cos(np.radians(self.geometries[self.EME,:]))
        DPHI = np.radians(self.geometries[self.PHI,:])
        R = np.radians(R)
        i = np.radians(self.geometries[self.INC,:])
        e = np.radians(self.geometries[self.EME,:])

        # Calculate CTHETA and THETA
        CTHETA = MU * MUP + np.sin(np.arccos(MU)) * np.sin(np.arccos(MUP)) * np.cos(DPHI)
        THETA = np.arccos(CTHETA)

        # Call Rness function
        MUP, MU, S = self.Rness(i, e, DPHI, R)

        # Calculate P
        P = (1 - CC) * (1 - BB**2) / ((1 + 2 * BB * CTHETA + BB**2)**(3 / 2))
        P += CC * (1 - BB**2) / ((1 - 2 * BB * CTHETA + BB**2)**(3 / 2))

        # Calculate B
        B = B0 * HH / (HH + np.tan(THETA / 2))

        # Calculate gamma, H0, and H
        gamma = np.sqrt(1 - W)
        H0 = (1 + 2 * MUP) / (1 + 2 * MUP * gamma)
        H = (1 + 2 * MU) / (1 + 2 * MU * gamma)

        # Calculate BRDF
        reflectances = W / 4 / (MU + MUP) * ((1 + B) * P + H0 * H - 1)
        reflectances = S * reflectances * MUP / np.cos(i)

        return reflectances

    def Rness(self,i, e, DPHI, R):
        mu0_e = np.zeros(len(i))
        mu_e = np.zeros(len(i))
        S = np.zeros(len(i))
    
        xidz = 1 / np.sqrt(1 + np.pi * np.tan(R)**2)
    
        mu_b = xidz * (np.cos(e) + np.sin(e) * np.tan(R) * self.e2(R, e) / (2 - self.e1(R, e)))
        mu0_b = xidz * (np.cos(i) + np.sin(i) * np.tan(R) * self.e2(R, i) / (2 - self.e1(R, i)))
    
        f = np.exp(-2 * np.tan(DPHI / 2))
    
        index1 = np.where(i <= e)
        index2 = np.where(i > e)
    
        if len(index1[0]) != 0:
            mu0_e[index1] = xidz * (np.cos(i[index1]) + np.sin(i[index1]) * np.tan(R) * (
                np.cos(DPHI[index1]) * self.e2(R, e[index1]) + np.sin(DPHI[index1] / 2)**2 * self.e2(R, i[index1])
            ) / (2 - self.e1(R, e[index1]) - (DPHI[index1] / np.pi) * self.e1(R, i[index1])))
        
            mu_e[index1] = xidz * (np.cos(e[index1]) + np.sin(e[index1]) * np.tan(R) * (
                self.e2(R, e[index1]) - np.sin(DPHI[index1] / 2)**2 * self.e2(R, i[index1])
            ) / (2 - self.e1(R, e[index1]) - (DPHI[index1] / np.pi) * self.e1(R, i[index1])))
        
            S[index1] = mu_e[index1] * np.cos(i[index1]) * xidz / mu_b[index1] / mu0_b[index1] / (
                1 - f[index1] + f[index1] * xidz * np.cos(i[index1]) / mu0_b[index1]
            )
    
        if len(index2[0]) != 0:
            mu0_e[index2] = xidz * (np.cos(i[index2]) + np.sin(i[index2]) * np.tan(R) * (
               self.e2(R, i[index2]) - np.sin(DPHI[index2] / 2)**2 * self.e2(R, e[index2])
            ) / (2 - self.e1(R, i[index2]) - (DPHI[index2] / np.pi) * self.e1(R, e[index2])))
        
            mu_e[index2] = xidz * (np.cos(e[index2]) + np.sin(e[index2]) * np.tan(R) * (
                np.cos(DPHI[index2]) * self.e2(R, i[index2]) + np.sin(DPHI[index2] / 2)**2 * self.e2(R, e[index2])
            ) / (2 - self.e1(R, i[index2]) - (DPHI[index2] / np.pi) * self.e1(R, e[index2])))
        
            S[index2] = mu_e[index2] * np.cos(i[index2]) * xidz / mu_b[index2] / mu0_b[index2] / (
                1 - f[index2] + f[index2] * xidz * np.cos(e[index2]) / mu_b[index2]
            )
    
        return mu0_e, mu_e, S

    def cot(self,x):
        sin_x = np.sin(x)
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-10
        sin_x = np.where(np.isclose(sin_x, 0, atol=epsilon), epsilon, sin_x)
        return np.cos(x) / sin_x

    def e1(self,R, x):
        return np.exp(-2 / np.pi * self.cot(R) * self.cot(x))

    def e2(self,R, x):
        return np.exp(-1 / np.pi * self.cot(R)**2 * self.cot(x)**2)

    def getDimensionY(self):
        return self.D_dimension

    def getDimensionX(self):
        return self.L_dimension

    def toPhysic(self, x):
        xp=np.copy(x)
        xp[0] = 1 - (1 -xp[0])**2
        for l in range(1,xp.shape[0]):
            xp[l] = xp[l] * self.scalingCoeffs[l] + self.offset[l]
        return xp

    def fromPhysic(self, x):
        xp=np.copy(x)
        xp[0] = 1 - np.sqrt(1 - xp[0])
        for l in range(1,xp.shape[0]):
            xp[l] = (xp[l] - self.offset[l]) / self.scalingCoeffs[l]
        return xp

    #################################################################################################
    ##                          OTHER FUNCTIONS (OPTIONAL)                                         ##
    #################################################################################################

    def __init__(self):
        # geometries data
        geom_tmp = []
        geom_tmp.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,40,40,40,40,40,40,40,40,40,40,40,40,40,40,60,60,60,60,60,60,60,60,60,60,60,60,60,60,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20])
        geom_tmp.append([70,60,50,40,30,20,10,10,20,30,40,50,60,70,70,60,50,40,30,20,10,0,10,30,40,50,60,70,70,60,50,30,20,10,0,10,20,30,40,50,60,70,70,60,50,40,30,20,10,0,10,20,30,40,50,70,70,60,50,40,30,20,10,0,10,20,30,40,50,60,70])
        geom_tmp.append([0,0,0,0,0,0,0,180,180,180,180,180,180,180,180,180,180,180,180,180,180,0,0,0,0,0,0,0,0,0,0,0,0,0,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,0,0,0,0,0,0,30,30,30,30,30,30,30,150,150,150,150,150,150,150,150])
        geometries=np.array(geom_tmp)
        self.D_dimension=geometries.shape[1]
        self.configuredGeometries=self.setupGeometries(geometries)
        self.geometries=geometries

    def setupGeometries(self, geometries):
        configuredGeometries = np.zeros(geometries.shape)
        geomsGrad = np.array(geometries)
        geomsGrad = np.radians(geomsGrad)
        return configuredGeometries
    
