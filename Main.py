# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:47:48 2022

@author: steffejb
"""

import os
from pyomo.common.tempfiles import TempfileManager
basepath = os.getcwd().replace(os.sep, '/')
TempfileManager.tempdir = basepath+"/temp/pyomo"
if not os.path.exists(basepath+"/temp/pyomo"):
    os.makedirs(basepath+"/temp/pyomo")
    

#os.chdir('C://Users//steffejb//OneDrive - NTNU//Work//GitHub//AIM_Norwegian_Freight_Model//AIM_Norwegian_Freight_Model')
#os.chdir("M:/Documents/GitHub/AIM_Norwegian_Freight_Model") #uncomment this for stand-alone testing of this fille
#os.chdir("C:/Users/Ruben/GitHub/STraM_ntnu_development") #uncomment this for stand-alone testing of this fille

from Model import TranspModel, RiskInformation
from ExtractResults import OutputData
from Data.ConstructData import TransportSets, get_scen_sheet_name
from Data.settings import *
from Data.interpolate import interpolate
from VisualizeResults import visualize_results
from ExtractModel import ModelExtractor

import pyomo.environ as pyo
from pyomo.environ import *
import time
import sys
original_stdout = sys.stdout
import pickle

import cProfile
import pstats

from Utils import Logger

#################################################
#                   user input                  #
#################################################

READ_DATA_FROM_FILE = False  #This can save some time in debug mode
analysis = "risk"  # ["standard","only_generate_data", "run_all2","single_time_period", "risk"]
scenario_tree = "FuelScen"     # Options: FuelScen,FuelDetScen, 4Scen, 9Scen, AllScen
analysis_type = "SP" #,   'EEV' , 'SP'         , expectation of expected value probem (EEV), stochastic program
co2_fee = "base" #"high, low", "base", "intermediate"
emission_cap_constraint = True   #False or True
wrm_strt = False  #use EEV as warm start for SP

store_solved_model = False


# risk parameters
cvar_coeff = 0.3    # \lambda: coefficient for CVaR in mean-CVaR objective
cvar_alpha = 0.8    # \alpha:  indicates how far in the tail we care about risk

log_to_file = True

time_periods = [2023, 2028, 2034, 2040, 2050]   # new time periods
num_first_stage_periods = 2         # how many of the periods above are in first stage
    

#################################################
#                   main code                   #
#################################################

def check_binding_capacity_constraints(model,EAT_INV_CONSTR_S):
    
    constraint_obj = getattr(model, "CapacitatedFlow", None)
    for (i,j,m,r,a,t,s) in EAT_INV_CONSTR_S:
        if m == "Rail":
            (ii,jj,mm,rr) = a
            index = (i,j,m,r,ii,jj,mm,rr,t,s)
            e = (i,j,m,r)
            #constr_name = f"CapacitatedFlow[{(i,j,m,r,a,t,s)}]"
            #constr = getattr(model, constr_name, None)
            constr_name = f"{a,t,s}"
            constr = constraint_obj[(i,j,m,r,a,t,s)]
            if constr is not None:
                lhs = pyo.value(constr.body)
                rhs = pyo.value(constr.lower) if constr.lower is not None else pyo.value(constr.upper)
                difference = rhs - lhs
                if difference == 0:
                    print(f"Constraint {a,t,s}, LHS: {lhs}, RHS: {rhs} BINDING.")
                else:
                    print(f"Constraint {a,t,s}, LHS: {lhs}, RHS: {rhs} not binding.")
                    #print(f"Constraint ({a,t,s}) not binding. Percentage difference = {round(difference/rhs*100,2)} %")
            else:
                print(f"Constraint {constr_name} not found in the model.")

def solve_init_model(base_data, risk_info,emission_cap_constraint):
    
    print("Solving the first time period for initialization purposes...")
            
    #update data
    time_periods = base_data.T_TIME_PERIODS
    base_data.T_TIME_PERIODS = [time_periods[0]]
    base_data.combined_sets()
    
    #
    model_instance_init = TranspModel(data=base_data, risk_info=risk_info)
    model_instance_init.emission_cap_constraint = emission_cap_constraint #does not really matter if the first year is 100%
    model_instance_init.construct_model()
    model_instance_init.solve_model(FeasTol=10**(-4),  #typically 10**(-6)
                        num_focus=0, # 0 is automatic, 1 is low precision but fast  https://www.gurobi.com/documentation/9.5/refman/numericfocus.html
                        #Method=-1,  
                        )

    #data back to normal        
    base_data.T_TIME_PERIODS = time_periods
    base_data.combined_sets()

    return model_instance_init

def construct_and_solve_model(base_data,
                                model_instance_init,
                                analysis_type,
                                risk_info, 
                                single_time_period=None,   #If not None, add year for which the analysis will be performed ("static" case)
                                NoBalancingTrips = None,
                                emission_cap_constraint = False,
                                FeasTol=None):
    # ------ CONSTRUCT MAIN MODEL ----------#
    print("-----")
    print(f"Constructing full {analysis_type} model...")

    start = time.time()
    model_instance = TranspModel(data=base_data, risk_info=risk_info)
    model_instance.NoBalancingTrips = NoBalancingTrips
    model_instance.single_time_period = single_time_period
    model_instance.emission_cap_constraint = emission_cap_constraint
    model_instance.construct_model()
    model_instance.fix_first_time_period(model_instance_init)
    print("Done constructing model.")
    print("Time used constructing the model:", time.time() - start)
    print("----------")
    sys.stdout.flush()

    #  ---------  SOLVE MODEL  ---------    #

    print("Solving model...",flush=True)
    start = time.time()
    model_instance.solve_model(FeasTol=FeasTol, #10**(-2),
                               #num_focus=1,
                               #Method=-1,  
                               ) 
    print("Done solving model.",flush=True)
    print("Time used solving the model:", time.time() - start,flush=True)
    print("----------", end="", flush=True)

    if False: #rail_analysis
        check_binding_capacity_constraints(model_instance.model,base_data.EAT_INV_CONSTR_S)

    return model_instance

def construct_and_solve_SP(base_data,
                            risk_info, 
                            single_time_period=None,   #If not None, add year for which the analysis will be performed ("static" case)
                            NoBalancingTrips = None,
                            emission_cap_constraint = False):
    

    # ------ CONSTRUCT and SOLVE INIT MODEL ----------#

    if True:   #We start with solving only the first time period. The output will be used to initialize the real problem.
        if single_time_period is None:
            model_instance_init = solve_init_model(base_data, risk_info,emission_cap_constraint)

    model_instance = construct_and_solve_model(base_data,
                                model_instance_init,
                                "SP",
                                risk_info, single_time_period,NoBalancingTrips,emission_cap_constraint,
                                FeasTol=10**-2)
    
    return model_instance,base_data

def construct_and_solve_EEV(base_data,risk_info,emission_cap_constraint=False):


        ############################
        ###  #2: solve EV        ###
        ############################
    
    #base_data.S_SCENARIOS = ['BBB']
    base_data.S_SCENARIOS = ['BB']
    base_data.combined_sets()

    
    # ------ CONSTRUCT and SOLVE INIT MODEL ----------#

    model_instance_init = solve_init_model(base_data, risk_info,emission_cap_constraint)
    
    # ------ CONSTRUCT MODEL ----------#

    model_instance_EV = construct_and_solve_model(base_data,
                                model_instance_init,
                                "EV",
                                risk_info, emission_cap_constraint,
                                FeasTol=10**-3)

    # --------- SAVE EV RESULTS -----------

    file_string = "EV_" + scenario_tree
    
    output = OutputData(model_instance_EV.model,base_data)

    with open(r"Data//output//" + file_string+'.pickle', 'wb') as output_file: 
        print("Dumping EV output in pickle file.....", end="",flush=True)
        pickle.dump(output, output_file)
        print("done.",flush=True)
    

        ############################
        ###  #3: solve EEV       ###
        ############################
    
    base_data.S_SCENARIOS = base_data.S_SCENARIOS_ALL
    base_data.combined_sets()

    # ------ CONSTRUCT MODEL ----------#

    print("Constructing EEV model...",end='', flush=True)

    start = time.time()
    model_instance = TranspModel(data=base_data, risk_info=risk_info)
    model_instance.construct_model()
    model_instance.fix_variables_first_stage(model_instance_EV.model)
    
    print("Done constructing EEV model.",flush=True)
    print("Time used constructing the model:", time.time() - start,flush=True)
    print("----------",  flush=True)


    #  ---------  SOLVE MODEL  ---------    #

    print("Solving EEV model...",end='',flush=True)
    start = time.time()
    #options = option_settings_ef()
    model_instance.solve_model(FeasTol=10**(-2),
                               #num_focus=1, 
                               #Method = -1,
                               ) 
    print("Done solving model.",flush=True)
    print("Time used solving the model:", time.time() - start,flush=True)
    print("----------",  flush=True)

    return model_instance, base_data

def construct_and_solve_SP_warm_start(base_data,
                            risk_info, 
                            ):
    
    model_instance, base_data = construct_and_solve_EEV(base_data,risk_info)

    model_instance.unfix_variables_first_stage()

    #  ---------  SOLVE MODEL  ---------    #

    print("Solving SP model with EEV warm start...",end='',flush=True)
    start = time.time()
    model_instance.solve_model(warmstart=True,FeasTol=10**(-2),num_focus=1)
    print("Done solving model.",flush=True)
    print("Time used solving the model:", time.time() - start,flush=True)
    print("----------", flush=True)

    return model_instance,base_data

def generate_base_data(scenario_tree,co2_fee="base",READ_FROM_FILE=False):
    
    identifier = scenario_tree+"_carbontax"+co2_fee+"_basedata"
    
    if READ_FROM_FILE:
        with open(r'Data//Output//'+identifier+'.pickle', 'rb') as data_file: 
            base_data = pickle.load(data_file)
    
    else:    

        sheet_name_scenarios = get_scen_sheet_name(scenario_tree)
        
        print("Reading data...", flush=True)
        start = time.time()
        base_data = TransportSets(sheet_name_scenarios=sheet_name_scenarios,co2_fee=co2_fee)                                # how many of the periods above are in first stage
        base_data = interpolate(base_data, time_periods, num_first_stage_periods)
        print("Done reading data.", flush=True)
        print("Time used reading the base data:", time.time() - start,flush=True)
        sys.stdout.flush()

        with open(r'Data//Output//'+identifier+'.pickle', 'wb') as data_file:   
            pickle.dump(base_data, data_file)
    
    return base_data

def save_solved_model(model_instance, run_identifier):
    model_extractor = ModelExtractor(model_instance.model,run_identifier).extract(run_identifier)

def main(scenario_tree,
         analysis_type,
         risk_aversion=None,
         cvar_coeff=cvar_coeff,
         cvar_alpha=cvar_alpha,
         single_time_period=None,
         NoBalancingTrips=False,
         co2_fee = "base",
         emission_cap=False
         ):
    
    #     --------- Setup  ---------   #

    sheet_name_scenarios = get_scen_sheet_name(scenario_tree)

    run_identifier = scenario_tree+"_carbontax"+co2_fee
    if emission_cap:
        run_identifier = run_identifier + "_emissioncap"
    if single_time_period is not None:
        run_identifier = run_identifier + "_stp"+str(single_time_period) 
    
    run_identifier2 = run_identifier+"_"+analysis_type
    if risk_aversion is not None:
        run_identifier2 = run_identifier2+"_"+risk_aversion

    current_logger = Logger(run_identifier2,log_to_file)
    sys.stdout = current_logger


    print('----------------------------')
    print('Doing the following analysis: ')
    print(run_identifier)
    if risk_aversion is not None:
        print(risk_aversion)
    if single_time_period is not None:
        print('single time period: ', single_time_period )
    print('----------------------------')
    #sys.stdout.flush()
    
    #     --------- DATA  ---------   #
    
    base_data = generate_base_data(scenario_tree, co2_fee=co2_fee,READ_FROM_FILE=READ_DATA_FROM_FILE)        

    if risk_aversion=="averse":
        cvar_alpha = 1-1/len(base_data.S_SCENARIOS)
    risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
    base_data.risk_information = risk_info
    
    if single_time_period is not None:
        base_data.single_time_period = single_time_period
        base_data.combined_sets()



    #     --------- MODEL  ---------   #
    # solve model
    if analysis_type == "SP":
        if wrm_strt:
            model_instance,base_data = construct_and_solve_SP_warm_start(base_data,risk_info)
        else:
            model_instance,base_data = construct_and_solve_SP(base_data,risk_info,single_time_period=single_time_period,emission_cap_constraint=emission_cap)
    elif analysis_type == "EEV":
        model_instance, base_data = construct_and_solve_EEV(base_data,risk_info)
    else:
        Exception('analysis type feil = '+analysis_type)
    
    #  --------- SAVE BASE DATA OUTPUT ---------    #
    
    print("Dumping data in pickle file...", end="")
    with open(r'Data//Output//'+run_identifier+'_basedata.pickle', 'wb') as data_file: 
        pickle.dump(base_data, data_file)
    print("done.")

    #-----------------------------------
    
    output = OutputData(model_instance.model,base_data)
    
    with open(r"Data//output//" + run_identifier2+'_results.pickle', 'wb') as output_file: 
        print("Dumping results in pickle file.....", end="")
        pickle.dump(output, output_file)
        print("done.")
        
    #  --------- STORE SOLVED MODEL ---------    #
        
    if store_solved_model:
        save_solved_model(model_instance, run_identifier2)

    #  --------- VISUALIZE RESULTS ---------    #

    if single_time_period is None:
        visualize_results(analysis_type,scenario_tree,
                            noBalancingTrips=NoBalancingTrips,
                            single_time_period=single_time_period,
                            risk_aversion=risk_aversion,
                            scen_analysis_carbon = False,
                            carbon_fee = co2_fee,
                            emission_cap=emission_cap
                        )
        
    sys.stdout.flush()
    sys.stdout = original_stdout
    current_logger.log.close()

def risk_analysis():
    for risk_avers in ["neutral","averse"]:
            if risk_avers == "neutral":
                cvar_coeff=0 #not being used
            elif risk_avers == "averse":
                cvar_coeff=0.99
                #cvar_alpha = 1/N
            main(scenario_tree,analysis_type="SP",cvar_coeff=cvar_coeff, risk_aversion=risk_avers)

if __name__ == "__main__":
    
    #import os
    #print(os.getcwd())
    
    if analysis == "only_generate_data":
        base_data = generate_base_data(scenario_tree, co2_fee=co2_fee,READ_FROM_FILE=READ_DATA_FROM_FILE)    
        
        risk_info = RiskInformation(cvar_coeff, cvar_alpha) # collects information about the risk measure
        base_data.risk_information = risk_info

        base_data.S_SCENARIOS = base_data.S_SCENARIOS_ALL
        
        base_data.combined_sets()

    elif analysis == "risk":
        risk_analysis()
    elif analysis == "standard":
        main(scenario_tree,analysis_type=analysis_type,co2_fee=co2_fee,emission_cap=emission_cap_constraint)
    elif analysis == "single_time_period":
        main(scenario_tree,analysis_type=analysis_type,single_time_period=2034)
        main(scenario_tree,analysis_type=analysis_type,single_time_period=2050)
    elif analysis == "carbon_price_sensitivity":
        for carbon_fee in ["low","high"]:
            main(scenario_tree,analysis_type="SP",co2_fee=carbon_fee)
    elif analysis=="run_all":
        main(scenario_tree,analysis_type="EEV",co2_fee=co2_fee)
        main(scenario_tree,analysis_type="SP",co2_fee=co2_fee)
        main(scenario_tree,analysis_type=analysis_type,single_time_period=2034)
        main(scenario_tree,analysis_type=analysis_type,single_time_period=2050)
        risk_analysis()
        for carbon_fee in ["low","high"]:
            main(scenario_tree,analysis_type="SP",co2_fee=carbon_fee)
    elif analysis=="run_all2":
        main("FuelScen",analysis_type=analysis_type,co2_fee="base",emission_cap=False)
        main("FuelScen",analysis_type=analysis_type,co2_fee="base",emission_cap=True)
        main("FuelScen",analysis_type=analysis_type,co2_fee="intermediate",emission_cap=False)
        main("FuelScen",analysis_type=analysis_type,co2_fee="high",emission_cap=False)
        #main("FuelScen",analysis_type=analysis_type,co2_fee="high",emission_cap=True)    

    # if profiling:
        #     profiler = cProfile.Profile()
        #     profiler.enable()






        


        
