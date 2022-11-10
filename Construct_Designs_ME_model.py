# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:08:44 2020

Using exact and heuristic algorithms to construct two-level QB-optimal designs
under the main effects model.

These are the results in Section 7.1.1 of the main text.
"""
import numpy as np # For storing arrays and matrix calculations.
from matfunctions import * # To evaluate the designs.
import time # To record time of the optimizations procedures.
from heuristic_algorithms import * # For coordinate exchange algorithm.
from exact_algorithms import * # For coordinate exchange algorithm.

#%% Select one design problem.
number_runs = 10 
number_factors = number_runs - 1

# Vector of probabilities.
pi_vec = [0.104, 0.188, 0.410, 0.625]

#%% Coordinate exchange (CE) algorithm.----------------------------------------
maxiter = 1000 # Maximum number of restarts.
time_CEalg = list()
Qbval_CEalg = list()
for pi_me in pi_vec:
    print("--- Probability: %s---" % pi_me)
    probs = compute_priors(number_factors, pi_me)
    start_time = time.time()
    CEdesign = CoordExch(number_runs, number_factors, probs.probs, maxiter)
    overalltime = time.time() - start_time
    time_CEalg.append(overalltime)
    Qbval_CEalg.append(CEdesign.Qb)

print('---RESULTS COORDINATE EXCHANGE ALGORITHM---')    
print('Computing times for maxiter iterations (s)')
print(time_CEalg)
print('Best Qval found after maxiter iterations')
print(Qbval_CEalg)
# Save objects.
save_fileCE = 'results/ME_'+ str(number_factors) +'_factors_CE_'+str(maxiter)+'iters_probs' + str(int(pi_me*1000)) + '.npz'
np.savez(save_fileCE, name1=Qbval_CEalg, name2=time_CEalg)

#%% Coordinate exchange (CE) algorithm.----------------------------------------
maxiter = 1000 # Maximum number of restarts.
time_RCP = list()
Qbval_RCP = list()
for pi_me in pi_vec:
    print("--- Probability: %s---" % pi_me)
    probs = compute_priors(number_factors, pi_me)
    start_time = time.time()
    RCPdesign = RCP(number_runs, number_factors, probs.probs, maxiter)
    overalltime = time.time() - start_time
    time_RCP.append(overalltime)
    Qbval_RCP.append(RCPdesign.Qb)  

print('---RESULTS RESTRICTED COLUMNWISE-PAIRWISE ALGORITHM---')    
print('Computing times for maxiter iterations (s)')
print(time_RCP)
print('Best Qval found after maxiter iterations')
print(Qbval_RCP)
# Save objects.
save_fileRCE = 'results/ME_'+ str(number_factors) +'_factors_RestColEx_'+str(maxiter)+'iters_probs' + str(int(pi_me*1000)) + '.npz'
np.savez(save_fileRCE, name1=Qbval_RCP, name2=time_RCP)

#%% Perturbation-Based Coordinate-Exchange (PBCE) Algorithm.---------------------
nrestarts = 5 # Number of restarts of the algorithm.
maxiter = 100 # Maximum number of iterations without an improvement.
perc_pert = 0.1 # Size of the perturbation.

# Solve ILP.
time_PBCE = list()
Qbval_PBCE = list()
for pi_me in pi_vec:
    print("--- Probability: %s---" % pi_me)
    probs = compute_priors(number_factors, pi_me)  
    start_time = time.time()
    PBCEdesign = PBCE(number_runs, number_factors, probs.probs, nrestarts, maxiter, perc_pert)
    overalltime = time.time() - start_time
    time_PBCE.append(overalltime)
    Qbval_PBCE.append(PBCEdesign.Qb)

print('---RESULTS ITERATED LOCAL SEARCH---')     
print('---Computing times for maxiter iterations (s)---')
print(time_PBCE)
print('---Best Qval found after maxiter iterations---')
print(Qbval_PBCE)
# Save objects.
pertlab = str(int(100*perc_pert))
save_filePBCE = 'results/'+ str(number_factors) +'_factors_PBCE_Repetitions'+ str(nrestarts)+'_'+str(maxiter)+'iters_Pert'+ pertlab+'_probs' + str(int(pi_me*1000)) + '.npz'
np.savez(save_filePBCE, name1=Qbval_PBCE, name2=time_PBCE)

#%% Point Exchange (PE) Algorithm.-----------------------------------------------------
maxiter = 100 # Maximum number of restarts.
if number_factors > 9:
    maxiter = 10

# Construct the candidate set.        
C = candidate_set(number_factors)

time_PE = list()
Qbval_PE = list()
for pi_me in pi_vec:
    print("--- Probability: %s---" % pi_me)
    probs = compute_priors(number_factors, pi_me)   
    start_time = time.time()
    PEdesign = PointExch(number_runs, C, probs.probs, maxiter)
    overalltime = time.time() - start_time
    time_PE.append(overalltime)
    Qbval_PE.append(PEdesign.Qb) 

print('---RESULTS POINT EXCHANGE ALGORITHM---')     
print('---Computing times for maxiter iterations (s)---')
print(time_PE)
print('---Best Qval found after maxiter iterations---')
print(Qbval_PE)

# Save objects.
save_filePE = 'results/ME_'+ str(number_factors) +'_factors_PE_'+str(maxiter)+'iters_probs' + str(int(pi_me*1000)) + '.npz'
np.savez(save_filePE, name1=Qbval_PE, name2=time_PE)

#%% Mixed Integer Quadratic Programming.---------------------------------------
max_search_time = 1200 # Maximum number of seconds for the Gurobi solver.
# Construct the candidate set.
C = candidate_set(number_factors)

gurobi_logfile = 'MIQP_QB_ME_Factors_' + str(number_factors) +'_factors_design.log'

time_MIQP = list()
Qbval_MIQP = list()
for pi_me in pi_vec:
    print("--- Probability: %s---" % pi_me)
    probs = compute_priors(number_factors, pi_me)   
    start_time = time.time()
    MIQPdesign = MIQP(number_runs, C, probs, t = max_search_time, 
                  verbose = True, n_replicates = 1, cnst_pre_overflow = 1000,
                   gurobi_log_name = gurobi_logfile)
    overalltime = time.time() - start_time
    time_MIQP.append(overalltime)
    Qbval_MIQP.append(MIQPdesign.Qb.Qbval) 

print('---RESULTS POINT EXCHANGE ALGORITHM---')     
print('---Computing times for maxiter iterations (s)---')
print(time_MIQP)
print('---Best Qval found after maxiter iterations---')
print(Qbval_MIQP)

save_fileMIQP = 'results/ME_'+ str(number_factors) +'_factors_MIQP.npz'
np.savez(save_fileMIQP, name1=Qbval_MIQP, name2=time_MIQP)
# To load results: data = np.load('mat.npz')