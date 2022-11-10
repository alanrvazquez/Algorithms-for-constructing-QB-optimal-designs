# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:08:44 2020

Using exact and heuristic algorithms to construct two-level QB-optimal designs
under the two-factor interaction model.

These are the results in Section 7.2.2 of the main text.
"""
import numpy as np # For storing arrays and matrix calculations.
from matfunctions import * # To evaluate the designs.
import time # To record time of the optimizations procedures.
from heuristic_algorithms import * # For coordinate exchange algorithm.
from exact_algorithms import * # For coordinate exchange algorithm.

#%% First set of problems: 7 factors.
number_factors = 7
number_runs = [16, 20, 24, 28, 32] 
pi_me = 0.5 # Prior probability for a main effect to be active.
pi_int_strong = 0.8 # Prior probability for a two-factor interaction to be active.

# Compute vector of probabilities.
probs = compute_priors(number_factors, pi_me, pi_int_strong)

#%% Second set of problems: 11 factors.
number_factors = 11
number_runs = [20, 24, 32, 40, 48] 
pi_me = 0.5 # Prior probability for a main effect to be active.
pi_int_strong = 0.4 # Prior probability for a two-factor interaction to be active.

# Compute vector of probabilities.
probs = compute_priors(number_factors, pi_me, pi_int_strong)

#%% Coordinate exchange (CE) algorithm.----------------------------------------
maxiter = 1000 # Maximum number of restarts.
time_CEalg = list()
Qbval_CEalg = list()
for nruns in number_runs:
    print("--- Number of runs: %s---" % nruns)
    start_time = time.time()
    CEdesign = CoordExch(nruns, number_factors, probs.probs, maxiter)
    overalltime = time.time() - start_time
    time_CEalg.append(overalltime)
    Qbval_CEalg.append(CEdesign.Qb)

print('---RESULTS COORDINATE EXCHANGE ALGORITHM---')    
print('Computing times for maxiter iterations (s)')
print(time_CEalg)
print('Best Qval found after maxiter iterations')
print(Qbval_CEalg)
# Save objects.
save_fileCE = 'results/'+ str(number_factors) +'_factors_CE_'+str(maxiter)+'iters.npz'
np.savez(save_fileCE, name1=Qbval_CEalg, name2=time_CEalg)

#%% Restricted columnwise-pairwise (RCP) algorithm.--------------------------------------
maxiter = 1000 # Maximum number of restarts.
time_RCP = list()
Qbval_RCP = list()
for nruns in number_runs:
    print("--- Number of runs: %s---" % nruns)
    start_time = time.time()
    RCPdesign = RCP(nruns, number_factors, probs.probs, maxiter)
    overalltime = time.time() - start_time
    time_RCP.append(overalltime)
    Qbval_RCP.append(RCPdesign.Qb)

print('---RESULTS RESTRICTED COLUMNWISE-PAIRWISE ALGORITHM---')    
print('Computing times for maxiter iterations (s)')
print(time_RCP)
print('Best Qval found after maxiter iterations')
print(Qbval_RCP)
# Save objects.
save_fileRCE = 'results/'+ str(number_factors) +'_factors_RestColEx_'+str(maxiter)+'iters.npz'
np.savez(save_fileRCE, name1=Qbval_RCP, name2=time_RCP)

#%% Point Exchange (PE) algorithm.-------------------------------------------------
maxiter = 100 # Maximum number of restarts.
if number_factors > 9:
    maxiter = 10

# Construct the candidate set.    
C = candidate_set(number_factors)

time_PE = list()
Qbval_PE = list()
for nruns in number_runs:
    print("--- Number of runs: %s---" % nruns)
    start_time = time.time()
    PEdesign = PointExch(nruns, C, probs.probs, maxiter)
    overalltime = time.time() - start_time
    time_PE.append(overalltime)
    Qbval_PE.append(PEdesign.Qb)

print('---RESULTS POINT EXCHANGE ALGORITHM---')     
print('---Computing times for maxiter iterations (s)---')
print(time_PE)
print('---Best Qval found after maxiter iterations---')
print(Qbval_PE)

# Save objects.
save_filePE = 'results/'+ str(number_factors) +'_factors_PE_'+str(maxiter)+'iters.npz'
np.savez(save_filePE, name1=Qbval_PE, name2=time_PE)


#%% Perturbation-Based Coordinate-Exchange (PBCE) Algorithm.--------------------------
nrestarts = 5 # Number of restarts of the algorithm.
maxiter = 100 # Maximum number of iterations without an improvement.
perc_pert = 0.1 # Size of the perturbation.

# Solve ILP.
time_PBCE = list()
Qbval_PBCE = list()
for nruns in number_runs:
    print("--- Number of runs: %s---" % nruns)
    start_time = time.time()
    PBCEdesign = PBCE(nruns, number_factors, probs.probs, nrestarts, maxiter, perc_pert)
    overalltime = time.time() - start_time
    time_PBCE.append(overalltime)
    Qbval_PBCE.append(PBCEdesign.Qb)

print('---RESULTS PERTURBATION-BASED COORDINATE-EXCHANGE ALGORITHM---')     
print('---Computing times for maxiter iterations (s)---')
print(time_PBCE)
print('---Best Qval found after maxiter iterations---')
print(Qbval_PBCE)
# Save objects.
pertlab = str(int(100*perc_pert))
save_filePBCE = 'results/'+ str(number_factors) +'_factors_PBCE_Repetitions'+ str(nrestarts)+'_'+str(maxiter)+'iters_Pert'+ pertlab+'.npz'
np.savez(save_filePBCE, name1=Qbval_PBCE, name2=time_PBCE)

#%% Mixed Integer Quadratic Programming.---------------------------------------
max_search_time = 1200 # Maximum search time for the solver.
gurobi_logfile = 'MIQP_QB_'+ str(number_factors) +'_factors.log'
# Construct the candidate set.
C = candidate_set(number_factors)

# Solve MIQP.
time_MIQP = list()
Qbval_MIQP = list()
for nruns in number_runs:
    print("--- Number of runs: %s---" % nruns)
    start_time = time.time()
    MIQPdesign = MIQP(nruns, C, probs, verbose = True, t = max_search_time, 
                       n_replicates = 1, gurobi_log_name = gurobi_logfile)
    overalltime = time.time() - start_time
    time_MIQP.append(overalltime)
    Qbval_MIQP.append(MIQPdesign.Qb.Qbval)

print('---RESULTS MIXED INTEGER QUADRATIC PROGRAMMING---')     
print('---Computing times for maxiter iterations (s)---')
print(time_MIQP)
print('---Best Qval found after maxiter iterations---')
print(Qbval_MIQP)

# Save objects.
save_fileMIQP = 'results/'+ str(number_factors) +'_factors_MIQP.npz'
np.savez(save_fileMIQP, name1=Qbval_MIQP, name2=time_MIQP)

# To load results: data = np.load('mat.npz')