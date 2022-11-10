# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:10:21 2020

Implementation of the Heuristic Algorithmic Approaches for
constructing two-level designs which optimize the Qb criterion.

"""

from matfunctions import *
import math

def Qboptimfun(D, w, nRuns, noF, nRunsq):
    """
    Function to compute the QB criterion value of a two-level design using the
    generalized word counts (calculated using Butlers moment matrix and
    updating formulas).
    
    Input:
        D: Design matrix with coded levels -1 and + 1 (numpy array).
        w: 5 x 1 vector of weights for the word counts (numpy array). 
        nRuns: run size of the design (int).
        noF: number of factors (int).
        nRunsq: squared of the run size of the design. (int)
       
    Output: 
        Approximation of Qb criterion value (float).
    """
    
    Tmat = np.matmul(D, D.T)
    Tmatflat = Tmat.ravel()
    Tmatflatsq = Tmatflat**2
    Mone = np.sum(Tmatflat)/nRunsq
    Mtwo = np.sum(Tmatflatsq)/nRunsq
    Mthree = Tmatflat.dot(Tmatflatsq)/nRunsq
    Mfour = Tmatflatsq.dot(Tmatflatsq)/nRunsq
     
    Qbval = w[0]*Mone + w[1]*Mtwo + w[2]*Mthree + w[3]*Mfour + w[4]
    
    return Qbval


def CoordExch(nRuns, noF, p, maxiter, myseed = 83301655, tol = 0.0000001):
    """
    Coordinate exchange algorithm to construct two-level QB efficient designs. 
    The algorithm uses the generalized word counts, calculated using Butlers moment matrix and
    updating formulas, and updated procedures for the moment matrix.
    
    Input:
        nRuns: Run size of the design (int)
		noF: Number of factors (int).
        p: 6 x 1 vector of prior probabilities for subsets of effects (numpy array). 
           That is p = [e10, e20, e21, e31, e32, e42].
        maxiter: Maximum number of iterations for the whole algorithm (int).
        myseed: set seed for reproducible results (int).
        tol: to cope with conditional statements involving float numbers. Also, 
             for reproducibility of results (float). 
       
    Output: 
        Best two-level Qb-efficient design found after 'maxiter' iterations of the
        algorithm (numpy array).
    """
    # Objects to save results.
    Qbvec = np.zeros((maxiter,1))
    designs = np.zeros((maxiter, nRuns, noF))
    
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
    
    for i in range(maxiter):
        # Generate starting design at random.
        Dcurr = 2*np.random.randint(2, size=(nRuns,noF))-1
        # Evaluate starting design. 
        Qbcurr = Qboptimfun(Dcurr, w, nRuns, noF, nRunssq)
        
        # Start coordinate exchange algorithm.------------------------------
        Qbpass = 10**10
        citer = 0
        while citer <= npass:
        
            for m in range(noF):
                                
                for l in range(nRuns):
                                    
                    # Switch the sign of coordinate.
                    Dcurr[l,m] = -1*Dcurr[l,m]
                    # Evaluate change.
                    Qbtest =  Qboptimfun(Dcurr, w, nRuns, noF, nRunssq)
                    
                    if (Qbtest - Qbcurr) < -tol: # If improvement.
                        Qbcurr = Qbtest # Update current best value of Qb criterion.
                    else :
                        # Restore the current best design and moment matrix.
                        Dcurr[l,m] = -1*Dcurr[l,m] # Switch the sign of coordinate back.
                        
            if abs(Qbpass - Qbcurr) > tol:
                Qbpass = Qbcurr
            else :
                break
          
            citer = citer + 1
            
        Qbvec[i] = Qbpass
        designs[i] = Dcurr    
      
    select_best = np.argmin(Qbvec)
    best_design = designs[select_best]
    return DesignClass(best_design, 0, np.min(Qbvec)/nRuns)

def contribution_point(Rmat, w, mFact, nRunssq):
    """
    Auxiliar function for Point exchange algorithm. The function calculates
    the contribution of a given row to the current design.
    
    Input:
        Rmat: Matrix including differences between a given row and
              and other rows included in the design (numpy array)
        w: 4 x 1 vecor of weights for the QB criterion (numpy array).
        mFact: vector of powers of number of factors (list). 
           That is [noF, noF**2, noF**3, noF**4].
        nRunssq: Number of runs to the power of two, nRuns**2 (int).
       
    Output: 
        Contribution to the QB criterion by a given row (float).
    """

    Rmatsq = Rmat**2
    Oone = (mFact[0] + 2*np.sum(Rmat))/nRunssq
    Otwo = (mFact[1] + 2*np.sum(Rmatsq))/nRunssq
    Othree = (mFact[2] + 2*Rmat.dot(Rmatsq))/nRunssq
    Ofour = (mFact[3] + 2*Rmatsq.dot(Rmatsq))/nRunssq
    row_contribution = w[0]*Oone + w[1]*Otwo + w[2]*Othree + w[3]*Ofour #+ w[4]
    return row_contribution


 
def PointExch(nRuns, X, p, maxiter = 1, n_replicates = 3, myseed = 83301655, tol = 0.0000001):
    """
    Point exchange algorithm to construct two-level QB efficient designs. 
    The algorithm uses the generalized word counts, calculated using Butlers moment matrix and
    updating formulas, and updated procedures for the moment matrix.
    Note: The algorithm uses a first improvement strategy.
    
    Input:
        nRuns: Run size of the design (int)
        X: N x noF candidate set, where noF is number of factors (numpy array).
        p: 6 x 1 vector of prior probabilities for subsets of effects (numpy array). 
           That is p = [e10, e20, e21, e31, e32, e42].
        maxiter: Maximum number of iterations for the whole algorithm (int).
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
    designs = np.zeros((maxiter, nRuns, noF))
    
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
        designs[i] = X[I,:]    
      
    select_best = np.argmin(Qbvec)
    best_design = designs[select_best]
    return DesignClass(best_design, 0, np.min(Qbvec)/nRuns)

def RCP(nRuns, noF, p, maxiter, myseed = 83301655, tol = 0.0000001):
    """
    Restricted columnwise-pairwise algorithm to construct balanced two-level QB efficient designs. 
    The algorithm uses the generalized word counts, calculated using Butlers moment matrix and
    updating formulas, and updated procedures for the moment matrix. The algorithm follows
    the implementation of Smucker and Drew (2015) "Approximate model spaces for model-robust 
    experiment design." Technometrics, 57:54-63.
    
    Input:
        nRuns: Run size of the design (int)
		noF: Number of factors (int).
        p: 6 x 1 vector of prior probabilities for subsets of effects (numpy array). 
           That is p = [e10, e20, e21, e31, e32, e42].
        maxiter: Maximum number of iterations for the whole algorithm (int).
        myseed: set seed for reproducible results (int).
        tol: to cope with conditional statements involving float numbers. Also, 
             for reproducibility of results (float). 
       
    Output: 
        Best two-level Qb-efficient design found after 'maxiter' iterations of the
        algorithm (numpy array).
    """
    
    Qbvec = np.zeros((maxiter,1))
    designs = np.zeros((maxiter, nRuns, noF))
    
    # Set seed.    
    np.random.seed(myseed)
    # Maximum number of times to pass.
    npass = 20*nRuns
    # Object allocation
    nRunssq = nRuns**2
    w = np.zeros((5,1))
    w[0] = p[0] + 2*(noF-1)*p[2] - (3*noF - 2)*p[3]
    w[1] = p[1] + p[2]/2 + (noF-2)*p[4] - (3*noF - 4)*p[5]/2 
    w[2] = p[3]
    w[3] = p[5]/4
    w[4] = noF*((3/4)*(noF-2)*p[5] - p[1] - p[2]/2 - (noF - 2)*p[4])
    nRunsdivtwo = int(nRuns/2)

    for i in range(maxiter):
    
        # Step 0: Generate starting design at random.
        Dcurr = np.ones((nRuns,noF))
        for f in range(noF):
            # Coordinates to change.
            row_idx = np.random.choice(nRuns, nRunsdivtwo, replace=False)
            Dcurr[row_idx,f] = -1*Dcurr[row_idx,f]
        # Evaluate starting design. 
        Qbcurr = Qboptimfun(Dcurr, w, nRuns, noF, nRunssq)
        # Start restricted column exchange algorithm.------------------------------
        Qbpass = 10**10
        citer = 0
        while citer <= npass:
        
            for m in range(noF):
                
                # Step 1: Randomnly choose an element of the column.
                coord_idx = np.random.choice(nRuns, 1, replace=False)[0]
                # Identify the elements with an oposite sign.
                oposite_sign = np.where(-Dcurr[coord_idx,m] == Dcurr[:,m])[0]
                # Switch the sign of the chosen element.
                Dcurr[coord_idx,m] = -1*Dcurr[coord_idx,m]
                
                # Step 2: Evaluate exchanges with all other elements of the column.
                Qb_exch = list()
                for l in oposite_sign:
                    # Switch the sign of element in turn.
                    Dcurr[l,m] = -1*Dcurr[l,m]
                    
                    # Evaluate change.
                    Qbtest = Qboptimfun(Dcurr, w, nRuns, noF, nRunssq)
                    Qb_exch.append(Qbtest[0])
                    
                    # Restore the element.
                    Dcurr[l,m] = -1*Dcurr[l,m] 
                    
                # Step 3: Test for improvement.
                Qb_exch_best = min(Qb_exch)
                if (Qb_exch_best - Qbcurr) < -tol: # If improvement.
                    Qbcurr = Qb_exch_best # Update current best value of Qb criterion.
                    best_ch = Qb_exch.index(min(Qb_exch))
                    best_coord = oposite_sign[best_ch]
                    # Switch the sign of the best coordinate.
                    Dcurr[best_coord,m] = -1*Dcurr[best_coord,m]      
                    
                else :
                    # Restore the current best design.
                    Dcurr[coord_idx,m] = -1*Dcurr[coord_idx,m] # Switch the sign of element back.
                        
            if abs(Qbpass - Qbcurr) > tol:
                Qbpass = Qbcurr
            else :
                break
          
            citer = citer + 1
            
            
        Qbvec[i] = Qbpass
        designs[i] = Dcurr    
      
    select_best = np.argmin(Qbvec)
    best_design = designs[select_best]
    return DesignClass(best_design, 0, np.min(Qbvec)/nRuns) 
 
def PBCE(nRuns, noF, p, nrestarts = 10, maxiter = 100, perc_pert = 0.1, myseed = 83301655, tol = 0.0000001):
    """
    Perturbation-Based Coordinate-Exchange (PBCE) algorithm to construct two-level QB efficient designs. 
    The algorithm uses the generalized word counts, calculated using Butlers moment matrix and
    updating formulas, and updated procedures for the moment matrix. The algorithm
    is based on a coordinate exchange strategy and uses perturbations to escape from
    local optimal solutions.
    
    Input:
        nRuns: Run size of the design (int)
		noF: Number of factors (int).
        p: 6 x 1 vector of prior probabilities for subsets of effects (numpy array). 
           That is p = [e10, e20, e21, e31, e32, e42].
        nrestarts: Maximum number of iterations for the whole algorithm (int).
        maxiter: Maximum number of iterations without an improvement after 
                 randomnly perturbing the incumbent solution (int).
        perc_pert: percentage of columns and rows to perturb (float).     
        myseed: set seed for reproducible results (int).
        tol: to cope with conditional statements involving float numbers. Also, 
             for reproducibility of results (float). 
       
    Output: 
        Best two-level Qb-efficient design found after 'nrestarts' iterations of the
        algorithm (numpy array).
    """
    
  # Objects to save results.
    Qbvec = np.zeros((nrestarts,1))
    designs = np.zeros((nrestarts, nRuns, noF))
    # Set seed.
    np.random.seed(myseed)
    # Maximum number of times to pass.
    npass = 20*nRuns
    # Size of perturbation.
    pert_nrows = math.ceil(nRuns*perc_pert)
    pert_ncols = math.ceil(noF*perc_pert)
    # Object allocation.
    nRunssq = nRuns**2
    mFact = [noF, noF**2, noF**3, noF**4]
    w = np.zeros((5,1))
    w[0] = p[0] + 2*(noF-1)*p[2] - (3*noF - 2)*p[3]
    w[1] = p[1] + p[2]/2 + (noF-2)*p[4] - (3*noF - 4)*p[5]/2 
    w[2] = p[3]
    w[3] = p[5]/4
    w[4] = noF*((3/4)*(noF-2)*p[5] - p[1] - p[2]/2 - (noF - 2)*p[4])
    
    for i in range(nrestarts):
        # Step 0: Generate starting design at random.
        Dcurr = 2*np.random.randint(2, size=(nRuns,noF))-1
        # Evaluate starting design. 
        Qbcurr = Qboptimfun(Dcurr, w, nRuns, noF, nRunssq)
        
        # Step 1. Optimize the starting design using coordinate exchange algorithm.
        Dcurr, Qbcurr = SingleCoordExchange(noF, nRuns, w, Dcurr, Qbcurr, npass, nRunssq, tol)
        # Save best design so far.
        designs[i] = Dcurr
        Qbvec[i] = Qbcurr
        # Step 2. Perturb the incumbent solution.
        nrep = 0
        while nrep <= maxiter:            
            # Perturb the best design.
            # Select the rows to perturb in terms of their contribution to the Qb criterion.
            rows_to_perturb = select_rows(designs[i], pert_nrows, nRuns, nRunssq, mFact, w) 
            cols_to_perturb = np.random.choice(noF, size = pert_ncols, replace = False)
            designs[i][np.ix_(rows_to_perturb,cols_to_perturb)] = -1*designs[i][np.ix_(rows_to_perturb,cols_to_perturb)]
            Qbcurr = Qboptimfun(designs[i], w, nRuns, noF, nRunssq)
            Dcurr, Qbcurr = SingleCoordExchange(noF, nRuns, w, designs[i].copy(), Qbcurr, npass, nRunssq, tol)

            if (Qbcurr - Qbvec[i]) < -tol:
                # If improvement, update the best solution.
                designs[i] = Dcurr
                Qbvec[i] = Qbcurr
                nrep = 0 # Start over again with perturbations.                
            else:
                # If no improvement, go back to original state.
                designs[i][np.ix_(rows_to_perturb,cols_to_perturb)] = -1*designs[i][np.ix_(rows_to_perturb,cols_to_perturb)]
                nrep = nrep + 1
    
    # Step 3: Select the best design after nrestarts.
    select_best = np.argmin(Qbvec)
    best_design = designs[select_best]
    return DesignClass(best_design, 0, np.min(Qbvec)/nRuns)      
    
def SingleCoordExchange(noF, nRuns, w, D, Qbval, npass, nRunssq, tol):
    """
    Simplified version of the coordinate exchange algorithm to construct 
    two-level QB efficient designs. 
    
    Input:
        noF: Number of factors (int).
        nRuns: Run size of the design (int)
        w: 5 x 1 vector of weights for the word counts (numpy array). 
        D: two-level design (numpy array).
        Qbval: Qb criterion value of design D (float).
        npass: Maximum number of times to pass through all the coordinates
               of the design without improvement (float).
        nRunssq: square of the run size (int)
        tol: to cope with conditional statements involving float numbers. Also, 
             for reproducibility of results (float). 
       
    Outputs: 
        D: Improved two-level design in terms of the Qb criterion value after
        one iteration of the algorithm (numpy array).
        Tmat: Butler's moment matrix of the improved design D (numpy array).
        Qbval: Qb criterion value of the improved design D (float).
    """
    # Start coordinate exchange algorithm.------------------------------
    Qbpass = 10**10
    citer = 0
    while citer <= npass:
    
        for m in range(noF):
                            
            for l in range(nRuns):
                                
                # Switch the sign of coordinate.
                D[l,m] = -1*D[l,m]
                # Evaluate change.
                Qbtest =  Qboptimfun(D, w, nRuns, noF, nRunssq)
                
                if (Qbtest - Qbval) < -tol: # If improvement.
                    Qbval = Qbtest # Update current best value of Qb criterion.
                else :
                    # Restore the current best design and moment matrix.
                    D[l,m] = -1*D[l,m] # Switch the sign of coordinate back.
                    
        if abs(Qbpass - Qbval) > tol:
            Qbpass = Qbval
        else :
            break
      
        citer = citer + 1
        
    return D, Qbval    


def select_rows(D, n, nRuns, nRunssq,  mFact, w):
    """
    Select the rows to perturb. 
    
    Input:
        D: two-level design (numpy array).
        n: Number of rows to select (int).
        nRuns: Run size of the design (int)
        nRunssq: square of the run size (int)
        mFact: vector of powers of number of factors (list). 
           That is [noF, noF**2, noF**3, noF**4].
        w: 5 x 1 vector of weights for the word counts (numpy array). 
        tol: to cope with conditional statements involving float numbers. Also, 
             for reproducibility of results (float). 
       
    Outputs: 
        Indices of the rows to perturb.
    """
    
    row_contribution = np.zeros((1,nRuns))
    nord = np.random.permutation(nRuns) # Shuffle the rows.
    Tmat = np.matmul(D[nord,:], D[nord,:].T)
    # Compute the individual contribution of each row.
    for i in range(nRuns):
        row_contribution[0,i] = contribution_point(np.delete(Tmat[i,:],i), w, mFact, nRunssq)
    
    rows_to_prt = row_contribution[0,:].argsort()[-n:][::-1]
    
    return nord[rows_to_prt]

