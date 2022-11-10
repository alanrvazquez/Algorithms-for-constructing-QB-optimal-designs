# -*- coding: utf-8 -*-
"""
Auxiliary functions and classes.

"""
import itertools 
import math # used by 'nchoosek' function
import numpy as np
import pickle
import operator as op
from functools import reduce
from pyDOE2 import fullfact # To generate candidate sets.

# Classes ---------------------------------------------------------------------
class ModMat(object):
    """
    Python class for model matrices.
    """
    def __init__(self, x, labels, relmat):
        self.x = x
        self.labels = labels
        self.relmat = relmat
        
class Qbclass(object):
    """
    Python class for Qb criterion value.
    """
    def __init__(self, Qbval, gwlp):
        self.Qbval = Qbval
        self.gwlp = gwlp

class DesignClass(object):
    """
    Python class for outputs of the MIQP function.
    """
    def __init__(self, design, z, Qb):
        self.design = design
        self.z = z
        self.Qb = Qb

class Probclass(object):
    """
    Python class for probabilites and data used in the MIQP approach.
    """
    def __init__(self, probs, weights, interactions):
        self.probs = probs
        self.weights = weights
        self.interactions = interactions         

# Functions -------------------------------------------------------------------
def save_object(obj, filename):
    """ Save object as pickle object in 'filename'.
    Input:
        obj: object (ModMat class, numpy array, etc.).
        filename: directory and name of the file (str).    
    """
    
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def nchoosek(n,r): 
    """ Compute the total number of combinations of 'n' in 'r'.
    """
    
    if n >= r and n >= 0 and r >= 0:
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        res = numer / denom
    else :
        res = 0
    return int(res)

def gencombintwo(n): 
    """ Create an array with all combinations of 'n' choose two.
    """
    nchtwo = nchoosek(n,2)
    ncombinations = np.zeros((nchtwo,2))
    c = 0
    for i, j in itertools.combinations(range(n),2):
        ncombinations[c,:] = i, j
        c = c + 1

    return ncombinations

def twofimodmat( D ):
    """ Construct two-factor interaction model matrix . 
    Input:
        D: matrix (numpy array, float).
    Output:
        Model matrix including main effect and two-factor 
        interaction columns, effect labels and heredity relationship matix 
        (ModMat class). 
    """
    
    (N,m) = np.shape(D)
    ncombtwo = itertools.combinations(range(m), 2) 
    nm = nchoosek(m, 2)
    Matt = np.zeros([N, nm])
    c = 0
    
    # Create relationship matrix
    fullsize = m+1
    fullsize = fullsize + nchoosek(fullsize-1,2)
    R = np.zeros((m+1,fullsize), dtype = int)
    R[0, 0:(m+1)] = np.ones((1, m+1), dtype = int)
    lec = m+2
    Combos = itertools.combinations(range(1, lec-1), 2)
    cc = lec - 1
    for i, j in Combos:
        R[i, cc] = 1
        R[j, cc] = 1
        cc = cc + 1 
    # Remove row and column for the intercept
    R = np.delete(R, 0, axis = 0)
    R = np.delete(R, 0, axis = 1)
    
    # Create labels for effects
    alphabet = []
    for letter in range(65,91):
        alphabet.append(chr(letter))
    # Labels for the MEs
    factors = alphabet[0:m]
    labels = alphabet[0:m] 
    
    # Construct two-factor interaction matrix
    for i,j in ncombtwo:
        Matt[:, c] = D[:, i]*D[:, j]
        # Labels for 2FIs
        lab_int = ':'.join([factors[i], factors[j]])
        labels.append(lab_int)  
        c = c + 1
    
    Intercolumn = np.ones((N,1))
    X = np.concatenate((D,Matt), axis = 1)
    X = np.concatenate((Intercolumn,X),axis=1)
    return ModMat(X, labels, R)

def GWLP(D):
    """
    Compute the generalized word counts of orders 1 up to 4 of a
    two-level design using the Butler's moment matrix.
    
    Input:
        D: two-level design with coded levels -1 and +1 (numpy array)
       
    Output: 
        List containing the B1-B4 generalized word counts.
    """
    nRuns, noF = np.shape(D)
    Tmat = np.matmul(D, D.T)
    Mone = np.sum(Tmat)/(nRuns**2)
    Mtwo = np.sum(Tmat**2)/(nRuns**2)
    Mthree = np.sum(Tmat**3)/(nRuns**2)
    Mfour = np.sum(Tmat**4)/(nRuns**2)
    
    Bone = Mone
    Btwo = (Mtwo - noF)/2
    Bthree = (Mthree - (3*noF-2)*Mone)/6
    Bfour = (Mfour - 2*(3*noF-4)*Mtwo + 3*noF*(noF-2))/24

    return([Bone, Btwo, Bthree, Bfour])  

def Qbtwolevel(D, p):
    """
    Compute the QB criterion value of a two-level design using the
    generalized word counts.
    
    Input:
        D: two-level design with coded levels -1 and +1 (numpy array)
        p: 6 x 1 vector of prior probabilities for subsets of effects (numpy array). 
       That is p = [e10, e20, e21, e31, e32, e42].
       
    Output: 
        Qbclass: Class which includes the Qb criterion value and the B1-B4
                 generalized word counts.
    """
    nRuns, noF = np.shape(D)
    Bone, Btwo, Bthree, Bfour = GWLP(D)
    Qbval = ((p[0] + 2*(noF-1)*p[2])*Bone + (2*p[1] + p[2] + 2*(noF-2)*p[4])*Btwo + 6*p[3]*Bthree + 6*p[5]*Bfour)/nRuns
    
    return Qbclass(Qbval[0], [Bone, Btwo, Bthree, Bfour])

def cfunprob(p_me, p_w, r, k = 1):
    """
    Compute the C_r in supplementary section C of Mee et al. (2017) "Selecting
    an orthogonal or non-orthogonal two-level design for screening". Technometrics
    59:305--318.
    
    Input:
        p_me: Prior probability for main effects (float).
        p_w: Prior probability for two-factor interactions following 
               weak effect heredity (float).
        r: Degree (int). 
        k: Exponent (int).
       
    Output: 
        C_r = [1 - pi_1 + pi_1(1 - pi_3)^r]^k.
    """
    return (1 - p_me + p_me*((1 - p_w)**r))**k


def priors_interactions(noF, p_me, p_int_s, p_int_w = 0):
    """
    Compute prior probabilities for the effects in the 
    two-factor interaction model.
    
    Input:
        noF: Number of factors (int).
        p_me: Prior probability for active main effects (float).
        p_int_s: Prior probability for active two-factor interactions 
                 following strong effect heredity (float).
        p_int_w: Prior probability for active two-factor interactions 
                 following weak effect heredity (float). 
       
    Output: 
        p: 6 x 1 vector of prior probabilities for subsets of effects (numpy array). 
           That is p = [e10, e20, e21, e31, e32, e42].
        w: Vector of weights for the MIQP optimization. 
    """  
    p = np.zeros((6,1))
    c_weak_one = 2*( (p_me)*(1 - p_me)*( 1 - (1 - p_int_w)*cfunprob(p_me, p_int_w, 1, noF - 2) )  ) + ( (1 - p_me)**2 )*(1 - 2*cfunprob(p_me, p_int_w, 1, noF - 2) + cfunprob(p_me, p_int_w, 2, noF - 2))
    c_weak_two = (p_me**2)*(1 - p_me)*p_int_s*(1 - ( (1 - p_int_w)**2 )*cfunprob(p_me, p_int_w, 1, noF - 3) ) + 2*( p_me*( (1 - p_me)**2 )*p_int_w*(1 - (1 - p_int_w)*cfunprob(p_me, p_int_w, 1, noF - 3) )) 
    c_weak_three = (p_me**2)*(1 - p_me)*(p_int_w**2) + 2*(p_me**2)*(1 - p_me)*p_int_w*p_int_s + p_me*( (1 - p_me)**2 )*(p_int_w**2)
    # Probability for ME
    p[0]  = p_me + (1 - p_me)*(1 - cfunprob(p_me, p_int_w, 1, noF - 1))
    # Probability for two MEs.
    p[1] = p_me**2 + c_weak_one
    # Probability for ME and INT with common factor
    p[2] = (p_me**2)*(p_int_s) + 2*p_me*(1 - p_me)*p_int_w
    # Probability ME and INT without common factor.
    p[3] = (p_me**3)*(p_int_s) + c_weak_two
    # Probability for interactions sharing a factor
    p[4] = (p_me**3)*(p_int_s**2) + c_weak_three
    p[5] = (p_me**4)*(p_int_s**2) # Interactions without common factor
    
    num_sets_two = nchoosek(noF,2)
    num_sets_three = nchoosek(noF,3)
    num_sets_four = nchoosek(noF,4)
    neffects = noF + num_sets_two + num_sets_three + num_sets_four  
    weights = np.zeros((neffects,1))
    c = 0
    for i in range(noF):
        weights[c] = p[0] + 2*(noF-1)*p[2]
        c = c+ 1
    
    # Matrix with two-factor interaction contrast vectors 
    for i, j in itertools.combinations(range(noF),2):
        weights[c] = 2*p[1] + p[2] + 2*(noF-2)*p[4]
        c = c+ 1
    
    # Matrix with three-factor interaction contrast vectors.
    for i, j, k in itertools.combinations(range(noF),3):
        weights[c] = 6*p[3]
        c = c+ 1
    
    # Matrix with four-factor interaction contrast vectors.    
    for i, j, k, l in itertools.combinations(range(noF),4):
        weights[c] = 6*p[5]
        c = c+ 1 

    return p, weights  

def priors_maineffects(noF, p_me):
    """
    Compute prior probabilities for the effects in a main effects model.
    
    Input:
        noF: Number of factors (int).
        p_me: Prior probability for main effects (float).
       
    Output: 
        p: 2 x 1 vector of prior probabilities for subsets of effects (numpy array). 
           That is p = [e10, e20].
        w: Vector of weights for the MIQP optimization. 
    """ 

    p = np.zeros((6,1))
    p[0]  = p_me # Probabilities for one ME
    p[1] = p_me**2 # Probabilities for two MEs
    p[2] = 0 
    p[3] = 0 
    p[4] = 0 
    p[5] = 0 
    
    num_sets_two = nchoosek(noF,2)
    neffects = noF + num_sets_two   
    weights = np.zeros((neffects,1))
    c = 0
    for i in range(noF):
        weights[c] = p[0] 
        c = c+ 1
    
    # Matrix with two-factor interaction contrast vectors 
    for i, j in itertools.combinations(range(noF),2):
        weights[c] = 2*p[1] 
        c = c+ 1

    return p, weights  

def compute_priors(noF, p_me, p_int_s = 0, p_int_w = 0):
    """
    Compute prior probabilities as specified by Mee et al. (2017).
    
    Input:
        noF: Number of factors (int).
        p_me: Prior probability for active main effects (float).
        p_int_s: Prior probability for active two-factor interactions 
                 following strong effect heredity (float).
        p_int_w: Prior probability for active two-factor interactions 
                 following weak effect heredity (float).         
    Output: 6 x 1 vector of prior probabilities for subsets of effects (numpy array). 
           That is p = [e10, e20, e21, e31, e32, e42], vector of weights for 
           the MIQP optimization, and whether the maximal model includes
           interactions (Probclass). 
    """    
    
    if p_int_s == 0: # If maximal model includes main effects only.
        p, weights = priors_maineffects(noF, p_me)
        inter = False
    else : # otherwise.
        p, weights = priors_interactions(noF, p_me, p_int_s, p_int_w)
        inter = True

    return Probclass(p, weights, inter)

def Essquared(X):
    """
    Compute E(s^2)-optimality criterion.
    
    Input:
        X: two-level design with coded levels -1 and +1 (numpy array).
       
    Output: 
        Esval: E(s^2)-optimality criterion (float). 
    """  

    k = np.shape(X)[1]
    sumcor = 0
    for combo in itertools.combinations(range(k), 2):  # 2 for pairs, 3 for triplets, etc
        sumcor = np.matmul(X[:,combo[0]].T, X[:,combo[1]])**2 + sumcor
    Esval = sumcor/nchoosek(k,2)
    return Esval

def candidate_set(noF):
    """
    Generate the candidate set.
    
    Input:
        noF: number of factors (int).
    Output: 
        X: two-level design with coded levels -1 and +1 (numpy array).
    """   
    # Generate full factorial design.
    levels = np.repeat(2,noF)
    X = np.asarray(2*fullfact(levels)-1)
    
    return X