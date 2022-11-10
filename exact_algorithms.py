# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:22:14 2019

Implementation of the Mixed Integer Quadratic Programming Algorithm for
constructing two-level Qb-optimal designs.

"""

import numpy as np # To perform matrix operations.
from gurobipy import * # To solve optimization problems.
from itertools import product, combinations # To use combinations and permutations.
from matfunctions import * # To evaluate the designs.
from heuristic_algorithms import * # For point exchange algorithm.
import math # For ceil and floor functions.

# Auxiliary functions.---------------------------------------------------------
def tolerance_integer(a, tol = 0.001):
    """ Transform a numeric value to a binary variable.
        This function corrects for possible misspecifications of the values
        of the integer decision variables resulting from the Gurobi Optimization.
    Input:
        a: variable value (int or float).
        tol: tolerance to be considered as integer (float).
    Output:
        zint: the integer value of a (int).
    """
    low_a = math.floor(a)
    up_a = math.ceil(a)
    
    if abs(a - up_a) < tol:
        zint = up_a
    elif abs(a - low_a) < tol:
        zint = low_a
    else :
        zint = a
        print('Decision variable z did not converge to an integer, ' + str(a))
    return(zint)    

def get_solution(model, N):
    """ Obtain the solutions from a gurobi model.
    Input:
        model: gurobi model.
        N: number of candidate points (int).
    Output:
        D: n x p numpy array containing the two-level design with coded levels
           -1 and +1.
    """
    Z = np.zeros(N)
    
    for i in range(N):
        zsol = model.getVarByName('z_'+ str(i))
        Z[i] = tolerance_integer(zsol.x)
            
    return Z

# Functions for MIQP approach.---------------------------------------------------------

def WarmStart(nRuns, N, noF, X, p, maxiter = 10, n_replicates = 3, myseed = 83301655, tol = 0.0000001):
    """
    Point exchange algorithm to generate warm starts for the MIQP problem.
    The algorithm uses the generalized word counts, calculated using Butlers moment matrix and
    updating formulas, and updated procedures for the moment matrix.
    Note: The algorithm uses a first improvement strategy.
    
    Input:
        nRuns: Run size of the design (int)
        X: N x noF candidate set, where noF is number of factors (numpy array).
        p: 6 x 1 vector of prior probabilities for subsets of effects (numpy array). 
           That is p = [e10, e20, e21, e31, e32, e42].
        maxiter: Maximum number of iterations for the whole algorithm (int).
        n_replicates: Maximum number of replicates of a point in the design (int). 
        myseed: set seed for reproducible results (int).
        tol: to cope with conditional statements involving float numbers. Also, 
             for reproducibility of results (float). 
       
    Output: 
        Best two-level Qb-efficient design found after 'maxiter' iterations of the
        algorithm (numpy array).
    """
    
    N, noF = np.shape(X)
    
    # Objects to save results.
    Qbvec = np.zeros((maxiter,1))
    designs = np.zeros((maxiter, nRuns), dtype='int')
    
    # Set seed.
    np.random.seed(myseed)
    # Maximum number of times to pass.
    npass = 20*nRuns
    
    # Object allocation.
    nRunssq = nRuns**2
    w = np.zeros((5,1))
    w[0] = p[0] + 2*(noF-1)*p[2] - (3*noF - 2)*p[3]
    w[1] = p[1] + p[2]/2 + (noF-2)*p[4] - (3*noF - 4)*p[5]/2 
    w[2] = p[3]
    w[3] = p[5]/4
    w[4] = noF*((3/4)*(noF-2)*p[5] - p[1] - p[2]/2 - (noF - 2)*p[4])
    mFact = [noF, noF**2, noF**3, noF**4]
    
    for i in range(maxiter):
        
        # Generate starting design at random.
        I = np.random.choice(range(N), nRuns, replace = False)
        
        # Start point exchange algorithm.------------------------------
        Qbpass = 10**10
        citer = 0
        while citer <= npass:
        
            for x_iter in range(nRuns):
                
                # Remove point from design.
                D_reduced = np.delete(X[I,:], x_iter, axis = 0)
                Rmat = np.matmul(X[ I[x_iter],:], D_reduced.T)
                QBp_curr = contribution_point(Rmat, w, mFact, nRunssq)
                
                other_points = np.delete(range(N), I[x_iter])                
                for y_iter in other_points:
                    
                    # Evaluate change.
                    Cand_Rmat = np.matmul(X[y_iter,:], D_reduced.T)              
                    QBp_test = contribution_point(Cand_Rmat, w, mFact, nRunssq)
                    rep_point = np.count_nonzero(I == y_iter)
                    
                    # If improvement and point is not repeated more than
                    #  'n_replicates' - 1 times.
                    if (QBp_test - QBp_curr) < -tol and (rep_point - n_replicates + 1) < tol: 
                        I[x_iter] = y_iter # Update design.
                        break # First improvement strategy.
            
            # Evaluate full design.
            Qbcurr = Qboptimfun(X[I,:], w, nRuns, noF, nRunssq)
            
            if abs(Qbpass - Qbcurr) > tol:
                Qbpass = Qbcurr
            else :
                break
          
            citer = citer + 1
            
        Qbvec[i] = Qbpass
        I.sort() # Sort indices.
        designs[i] = I    
    
    # Select best solution after 'maxiter' iterations.
    select_best = np.argmin(Qbvec)
    best_design = designs[select_best]

    # Turn best solution into a warm start.
    warm_sol = np.zeros(N, dtype='int')
    for i in range(nRuns):
        warm_sol[ best_design[i] ] = warm_sol[ best_design[i] ] + 1 
        
    return(warm_sol) 

def construct_design(zsol, X, N, m):
    """ Construct design matrix from solution to MIQP problem.
    Input:
        zsol: integer solution to MIQP program (numpy array).  
        X: N x m candidate set including (numpy array).
        N: number of candidaet points (int).
        m: number of factors (int).
    Output:
        D: n x m numpy array containing the two-level design with coded levels
           -1 and +1. The value of n is the sum of all elements in zsol.
    """
    design = np.zeros((1,m))
    for i in range(N):
        candidate_point = X[i,:].reshape((1,m))
        if zsol[i] > 0:
            new_rows = np.repeat(candidate_point, repeats = zsol[i], axis = 0)
            design = np.vstack((design, new_rows))    
    D = design[1:,:]
    return(D)

def MIQP(n, X, probs, advanced = False, n_replicates = 2, t = 60, opt_focus = 0, symmetry = -1, maxiter = 10, verbose = False, cnst_pre_overflow = 1, gurobi_log_name = "MIQP_QB_problems.log"):
    """ Find two-level Q_b-optimal designs using mixed integer quadratic
        programming (MIQP).
    Input:
        n: number of runs (int).
        X: N x k candidate set including (numpy array).
        probs: 6 x 1 vector of prior probabilities for subsets of effects (numpy array). 
           That is p = [e10, e20, e21, e31, e32, e42].
        advanced: Use the point-exchange algorithm to provide a warm start (boolean).
        n_replicates: maximum number of replicates allowed in the design (int).
        t: maximum search time for the Gurobi solver (int).
        opt_focus: parameter of the Gurobi solver which sets the focus of the solver (int).
        symmetry: parameter of the Gurobi solver which sets the strategy to deal with symmetry (int).
        maxiter: number of iterations of the point exchange algorithm (int). 
        cnst_pre_overflow: constant to prevent overflow in Gurobi when dealing with very small prior probabilities (int). 
        gurobi_log_name: name of the Gurobi log file (str). 

    Output:
        DesignClass object
            - Qboptim_design: n x p numpy array containing the two-level design with coded levels
              -1 and +1 (numpy array).
            - z: 1 x N vector of binary decision variables (numpy array).
            - Qbval: Qb-optimality criterion value.  
    """
  
    # PREPROCESSING=======================================================    
    N, m = np.shape(X)
    normalizer = N*n_replicates
    # Matrix with two-factor interaction contrast vectors
    num_sets_two = nchoosek(m,2) 
    Mtwo = np.zeros((N, num_sets_two))
    c = 0
    for i, j in itertools.combinations(range(m),2):
        Mtwo[:,c] = X[:,i]*X[:,j]
        c = c+ 1
    # Total number of parameters.    
    p = m + num_sets_two  

    if probs.interactions :
        num_sets_three = nchoosek(m,3)
        # Matrix with three-factor interaction contrast vectors.
        Mthree = np.zeros((N, num_sets_three))
        c = 0
        for i, j, k in itertools.combinations(range(m),3):
            Mthree[:,c] = X[:,i]*X[:,j]*X[:,k]
            c = c+ 1
        # Matrix with four-factor interaction contrast vectors.    
        num_sets_four = nchoosek(m,4)
        Mfour = np.zeros((N, num_sets_four))
        c = 0
        for i, j, k, l in itertools.combinations(range(m),4):
            Mfour[:,c] = X[:,i]*X[:,j]*X[:,k]*X[:,l]
            c = c+ 1        
        # Total number of parameters.    
        p = p + num_sets_three + num_sets_four     
    
    #==COMPUTE WARM START==================================================
    if advanced:
        if verbose:
            print("Creating initial design...")
        warm_sol = WarmStart(n, N, m, X, probs.probs, maxiter)
        if verbose:
            sel_points = abs(warm_sol - 1) < 0.0000001
            b = Qbtwolevel(X[sel_points,:], probs.probs)
            print("Initial design created with QB criterion value: ", b.Qbval)
    
    # OPTIMIZATION PROBLEM ================================================
    model_name = "MIQP_QB_Model"
    
    # Set MIQP model parameters.
    model = Model(model_name)
    model.params.outputflag = 0
    #model.params.MIPGap = 0.00000001
    if verbose:
        model.params.outputflag = 1
        model.params.LogFile = gurobi_log_name
    model.params.MIPfocus = opt_focus
    model.params.Symmetry = symmetry    
    
    
    #==CREATE VARIABLES====================================================
    # Continuous variables.
    y_vecvar = []
    for i in range(p):
        def_cont_var = model.addVar(lb=-1,ub=1,vtype=GRB.CONTINUOUS, name = "y_{0}".format(i))
        y_vecvar.append(def_cont_var)
    # Integer variables.    
    z_vecvar = []
    for i in range(N):
        def_int_var = model.addVar(lb=0, ub=n_replicates, vtype=GRB.INTEGER, name = "z_{0}".format(i))
        z_vecvar.append(def_int_var)  
        
    model.update()
    
    #==DEFINE AND ADD CONSTRAINTS==========================================
    c = 0
    for i in range(m):
        model.addConstr(y_vecvar[c] == (1/normalizer)*quicksum(X[l,i]*z_vecvar[l] for l in range(N)), name = "constfor_ME_"+str(i))
        c = c + 1    
    
    for i in range(num_sets_two):
        model.addConstr(y_vecvar[c] == (1/normalizer)*quicksum(Mtwo[l,i]*z_vecvar[l] for l in range(N)), name = "constfor_TWOFI_"+str(i))
        c = c + 1    
    if probs.interactions : 
        for i in range(num_sets_three):
            model.addConstr(y_vecvar[c] == (1/normalizer)*quicksum(Mthree[l,i]*z_vecvar[l] for l in range(N)), name = "constfor_THREEFI_"+str(i))
            c = c + 1    
        
        for i in range(num_sets_four):
            model.addConstr(y_vecvar[c] == (1/normalizer)*quicksum(Mfour[l,i]*z_vecvar[l] for l in range(N)), name = "constfor_FOURFI_"+str(i))
            c = c + 1        
    
    # Run size of the design.
    model.addConstr(sum(z_vecvar) == n, name = 'run_size')  
    
    #==SET OBJECTIVE FUNCTION==============================================
    sumsq_yis = quicksum(probs.weights[i][0]*y_vecvar[i]*y_vecvar[i] for i in range(p)) 
    model.setObjective(cnst_pre_overflow*sumsq_yis, GRB.MINIMIZE)        
    model.update() 
    
    #==SET TIME LIMIT====================================================== 
    model.params.timeLimit = t

    #==SET WARM STARTS==================================================
    if advanced:            
        for i in range(N):
            z_vecvar[i].start = warm_sol[i]     
    
    #==OPTIMIZE MODEL======================================================
    model.optimize()       
    
    # Retrieve results
    zsol = get_solution(model, N)
    Qboptim_design = construct_design(zsol, X, N, m)
    Qbval = Qbtwolevel(Qboptim_design, probs.probs)

    return DesignClass(Qboptim_design, zsol, Qbval)