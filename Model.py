# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:52:48 2022

@author: steffejb
"""

#Pyomo
import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
import pyomo.opt.base as mobase
from pyomo.environ import *
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
import logging

import logging
from Data.settings import *


############# Class ################

class RiskInformation:

    def __init__(self, cvar_coeff, cvar_alpha):
        self.cvar_coeff = cvar_coeff    # \lambda: coefficient for CVaR in the mean-cvar objective
        self.cvar_alpha = cvar_alpha    # \alpha:  indicates how far in the tail we care about risk (0.9 means we care about worst 10%)

class TranspModel:

    def __init__(self, data, risk_info):

        self.model = ConcreteModel()
        
        self.data = data
        self.risk_info = risk_info # stores parameters of the risk measure (i.e., lambda and alpha for mean-CVaR)

        self.calculate_max_transp_amount_exact = False

        self.solve_base_year = False
        self.single_time_period = None
        self.emission_cap_constraint = False

        self.NoBalancingTrips = False

    def construct_model(self,risk_neutral=False):

        "VARIABLES"
        #def fb(model, i,j,m,r,f,p,,t,s):
        #    return (lb[i], ub[i])
        #self.model.x_flow = Var(self.data.AFPT_S, within=NonNegativeReals,bounds=fb)
        self.model.x_flow = Var(self.data.AFPT_S, within=NonNegativeReals)
        self.model.b_flow = Var(self.data.AFVT_S, within=NonNegativeReals)
        self.model.h_path = Var(self.data.KPT_S, within=NonNegativeReals)# flow on paths K,p
        self.model.h_path_balancing = Var(self.data.KVT_S, within=NonNegativeReals,initialize=1)# flow on paths K,p
        if self.NoBalancingTrips:
            for (i,j,m,r,f,v,t,s) in self.data.AFVT_S:
                a = (i,j,m,r)
                self.model.b_flow[(a,f,v,t,s)].fix(0)
            for (k,v,t) in self.data.KVT_S:
                self.model.h_path_balancing[(k,v,t,s)].fix(0)
        self.model.StageCosts = Var(self.data.T_TIME_PERIODS_S, within = NonNegativeReals)
        
        self.model.epsilon_edge = Var(self.data.ET_INV_S, within = Binary) #within = Binary
        self.model.upsilon_upg = Var(self.data.UT_UPG_S, within = Binary) #bin.variable for investments upgrade/new infrastructure u at link l, time period t
        def contin01(model, i,m,t,s): #()
            return (0, 1)
        self.model.nu_node = Var(self.data.NM_CAP_INCR_T_S, within = NonNegativeReals, bounds=contin01) #investment in terminals (continuous)
        self.model.y_charge = Var(self.data.EFT_CHARGE_S, within=NonNegativeReals)
        
        self.model.q_transp_amount = Var(self.data.MFT_S, within=NonNegativeReals)
        self.model.q_transp_delta = Var(self.data.MFT_MIN0_S, within=NonNegativeReals)

        # auxiliary transport amount for tracking diffusion paths:
        self.model.q_aux_transp_amount = Var(self.data.MFT_NEW_YEARLY_S, within = NonNegativeReals) 
        # total transport amount per mode
        self.model.q_mode_total_transp_amount = Var(self.data.MT_S, within = NonNegativeReals) #note: only measured in decision years
        
        #if EMISSION_CONSTRAINT:
        self.model.total_emissions = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        self.model.emissions_penalty = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)

        #self.model.feas_relax = Var(within=NonNegativeReals)  
        #self.model.feas_relax.setlb(0)
        #self.model.feas_relax.setub(0)

        #COST_VARIABLES


        self.model.TranspOpexCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TranspOpexCost(model, t,s):
            return (self.model.TranspOpexCost[t,s] >= sum((self.data.C_TRANSP_COST[(i,j,m,r,f,p,t,s)])*self.model.x_flow[(i,j,m,r,f,p,t,s)] 
                                      for p in self.data.P_PRODUCTS for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m])- FEAS_RELAX )  
        self.model.TranspOpexCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TranspOpexCost)
        
        self.model.TranspCO2Cost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TranspCO2Cost(model, t,s):
            return (self.model.TranspCO2Cost[t,s] >= sum((self.data.C_CO2[(i,j,m,r,f,p,t)])*self.model.x_flow[(i,j,m,r,f,p,t,s)] 
                                      for p in self.data.P_PRODUCTS for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) -FEAS_RELAX )  
        self.model.TranspCO2CostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TranspCO2Cost)

        self.model.CO2_PENALTY = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def CO2Pen(model, t,s):
            co2_fee_penalty = 10000 #NOK/tonneCO2
            co2_fee_penalty_adj = co2_fee_penalty/(1000*1000)   #NOK/gCO2
            co2_fee_penalty_adj = co2_fee_penalty_adj/SCALING_FACTOR_MONETARY*SCALING_FACTOR_EMISSIONS
            return (self.model.CO2_PENALTY[t,s] >= co2_fee_penalty_adj*self.model.emissions_penalty[(t,s)]-FEAS_RELAX )  
        self.model.CO2PenaltyConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=CO2Pen)
        
        self.model.TranspOpexCostB = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TranspOpexCostB(model, t,s):
            return (self.model.TranspOpexCostB[t,s] >= sum( EMPTY_VEHICLE_FACTOR*(self.data.C_TRANSP_COST[(i,j,m,r,f,self.data.cheapest_product_per_vehicle[(m,f,t,v)],t,s)]) * self.model.b_flow[(i,j,m,r,f,v,t,s)] 
                                    for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] ) - FEAS_RELAX )  
        self.model.TranspOpexCostBConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TranspOpexCostB)
        
        self.model.TranspCO2CostB = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TranspCO2CostB(model, t,s):
            return (self.model.TranspCO2CostB[t,s] >= sum( EMPTY_VEHICLE_FACTOR*(self.data.C_CO2[(i,j,m,r,f,self.data.cheapest_product_per_vehicle[(m,f,t,v)],t)]) * self.model.b_flow[(i,j,m,r,f,v,t,s)] 
                                    for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] ) -FEAS_RELAX)  
        self.model.TranspCO2CostBConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TranspCO2CostB)

        self.model.TranspTimeCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TranspTimeCost(model, t, s):
            return (self.model.TranspTimeCost[t,s] >= sum( self.data.C_TIME_VALUE[(i,j,m,r,p)] * self.model.x_flow[(i,j,m,r,f,p,t,s)] 
                                                          for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for p in self.data.P_PRODUCTS)  - FEAS_RELAX)
        self.model.TranspTimeCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule = TranspTimeCost)

        self.model.TransfCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def TransfCost(model, t,s):
            return (self.model.TransfCost[t,s] >= sum((self.data.C_TRANSFER[(k,p)]+self.data.C_TRANSFER_TIME[(k,p)])*self.model.h_path[k,p,t,s] for p in self.data.P_PRODUCTS  for k in self.data.PATHS_NO_UNIMODAL_ROAD)-FEAS_RELAX )  
        self.model.TransfCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=TransfCost)

        
        self.model.EdgeCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def EdgeCost(model, t,s):
            return (self.model.EdgeCost[t,s] >= sum(self.data.C_EDGE_INV[e]*self.model.epsilon_edge[(e,t,s)] for e in self.data.E_EDGES_INV if e +(t,) in self.data.ET_INV)- FEAS_RELAX )  
        self.model.EdgeCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=EdgeCost)
        
        self.model.NodeCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def NodeCost(model, t,s):
            return (self.model.NodeCost[t,s] >= sum(self.data.C_NODE_INV[(i,m)]*self.model.nu_node[(i,m,t,s)] for (i, m) in self.data.NM_CAP_INCR 
                                                    if (i,m,t) in self.data.NM_CAP_INCR_T)- FEAS_RELAX )  
        self.model.NodeCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=NodeCost)
        
        self.model.UpgCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def UpgCost(model, t,s):
            return (self.model.UpgCost[t,s] >= sum(self.data.C_EDGE_UPG[(e,f)]*self.model.upsilon_upg[(e,f,t,s)] for (e,f) in self.data.U_UPGRADE if (e,f,t) in self.data.UT_UPG) - FEAS_RELAX)  
        self.model.UpgCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=UpgCost)
        
        self.model.ChargeCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def ChargeCost(model, t,s):
            return (self.model.ChargeCost[t,s] >= sum(self.data.C_CHARGE[(e,f,t)]*self.model.y_charge[(e,f,t,s)] 
                                                      for (e,f) in self.data.EF_CHARGING 
                                                      if f=="Battery")
                                                      - FEAS_RELAX )
        self.model.ChargeCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=ChargeCost)

        self.model.FillingCost = Var(self.data.T_TIME_PERIODS_S, within=NonNegativeReals)
        def FillingCost(model, t,s):
            return (self.model.FillingCost[t,s] >= sum(self.data.C_CHARGE[(e,f,t)]*self.model.y_charge[(e,f,t,s)] 
                                                       for (e,f) in self.data.EF_CHARGING 
                                                      if f=="Hydrogen")- FEAS_RELAX )
        self.model.FillingCostConstr = Constraint(self.data.T_TIME_PERIODS_S, rule=FillingCost)
    

        # CVaR variables
        
        # CVaR auxiliary variable (corresponds to u)
        self.model.CvarAux = Var(within=Reals)

        #CVaR positive part variable (corresponds to z = (f - u)^+)
        self.model.CvarPosPart = Var(self.data.S_SCENARIOS, within=NonNegativeReals)               #note: includes one positive part constraint: z \geq 0


        "OBJECTIVE"
        #-------------------------

        def StageCostsVar(model, t,s):  

            # Pyomo SUM_PRODUCT is slower than the following
            yearly_transp_cost = (self.model.TranspOpexCost[t,s] + 
                                  self.model.TranspCO2Cost[t,s] +
                                  self.model.CO2_PENALTY[t,s] +
                                  self.model.TranspOpexCostB[t,s] + 
                                  self.model.TranspCO2CostB[t,s] + 
                                  self.model.TranspTimeCost[t,s] + 
                                  self.model.ChargeCost[t,s] + #ANNUAL CHARGING/FILLING INFRASTR. COST INCLUDED HERE
                                  self.model.FillingCost[t,s] + #ANNUAL CHARGING/FILLING INFRASTR. COST INCLUDED HERE
                                  0) 
            
            if t == self.data.T_TIME_PERIODS[-1]:
                #factor = discount back from the last time period * infinite cash flow stream
                #factor = round(self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0] * (1/(1-self.data.D_DISCOUNT_RATE)),self.data.precision_digits)      #was 2.77, becomes 9
                #alternatively, use simply 10 years
                factor = round(sum(self.data.D_DISCOUNT_RATE**n for n in self.data.Y_YEARS[t]),self.data.precision_digits)
            else:
                factor = round(sum(self.data.D_DISCOUNT_RATE**n for n in self.data.Y_YEARS[t]),self.data.precision_digits)
            opex_costs = factor*(yearly_transp_cost+self.model.TransfCost[t,s]) 
            
            delta = round(self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0],self.data.precision_digits)
            investment_costs = self.model.EdgeCost[t,s] + self.model.NodeCost[t,s] + self.model.UpgCost[t,s]  
            
            return (self.model.StageCosts[t,s] >= (opex_costs + delta*investment_costs ) - FEAS_RELAX ) 
        
        self.model.stage_costs = Constraint(self.data.T_TIME_PERIODS_S, rule = StageCostsVar)
        
     
        # first-stage objective value variable
        self.model.FirstStageCosts = Var(self.data.S_SCENARIOS, within = Reals)
        def FirstStageCostsRule(model,s):
            return (self.model.FirstStageCosts[s] >= sum(self.model.StageCosts[t,s] for t in self.data.T_TIME_PERIODS if t in self.data.T_TIME_FIRST_STAGE) - FEAS_RELAX ) 
        self.model.FirstStageCostsConstr = Constraint(self.data.S_SCENARIOS, rule = FirstStageCostsRule)

        # second-stage objective value variable
        self.model.SecondStageCosts = Var(self.data.S_SCENARIOS, within = Reals)
        def SecondStageCostsRule(model,s):
            return (self.model.SecondStageCosts[s] >= sum(self.model.StageCosts[t,s] for t in self.data.T_TIME_PERIODS if t in self.data.T_TIME_SECOND_STAGE) - FEAS_RELAX)
        self.model.SecondStageCostsConstr = Constraint(self.data.S_SCENARIOS,rule = SecondStageCostsRule)

        # scenario objective value variable (risk-neutral model would take expectation over this in the objective)
        #self.model.ScenObjValue = Var(within=Reals)
        #def ScenObjValue(model):
        #    return self.model.ScenObjValue == sum(self.model.StageCosts[t] for t in self.data.T_TIME_PERIODS) +  self.model.MaxTranspPenaltyCost  # corresponds to f(x,\xi)
        #self.model.ScenObjValueConstr = Constraint(rule = ScenObjValue)
        
        #CVaR positive part constraint
        def CvarRule(model,s):
            return (self.model.CvarPosPart[s] >= self.model.SecondStageCosts[s] - self.model.CvarAux - FEAS_RELAX  )     # z \geq v - u
        self.model.CvarPosPartConstr = Constraint(self.data.S_SCENARIOS, rule=CvarRule) # add to model

        # lower bound on auxiliary CVaR variable (needed to avoid numerical issues)
        cvar_aux_lb = 0.0 # (hardcoded). 10 seems to be more accurate than 11
        def CvarAuxLBRule(model):
            return (self.model.CvarAux >= cvar_aux_lb )
        self.model.CvarAuxLBConstr = Constraint(rule=CvarAuxLBRule) # add to model

        # mean-CVaR:
        def objfun_risk_averse(model):          
            # scenario objective value for risk-averse (mean-CVaR) model:
            mean_cvar_scen_obj_value = 1/len(self.data.S_SCENARIOS)*sum(self.model.FirstStageCosts[s] + self.risk_info.cvar_coeff * self.model.CvarAux + (1 - self.risk_info.cvar_coeff) * self.model.SecondStageCosts[s] + self.risk_info.cvar_coeff / (1 - self.risk_info.cvar_alpha) * self.model.CvarPosPart[s] for s in self.data.S_SCENARIOS)
            #                          ####### c*x ##############   ############# lambda * u #####################   ####################### (1 - lambda) * q*y ##################   ############### lambda / (1 - alpha) * q*y #########################################                                                             
            return mean_cvar_scen_obj_value

        # # pure CVaR:
        # def objfun_pure_cvar(model):          
        #     # scenario objective value for risk-averse (mean-CVaR) model:
        #     cvar_scen_obj_value = self.model.FirstStageCosts + self.model.CvarAux + 1.0/(1 - self.risk_info.cvar_alpha) * self.model.CvarPosPart  
        #     return cvar_scen_obj_value

        # # risk-neutral #NOTE: TEMPORARY
        def objfun_risk_neutral(model):          
            # scenario objective value for risk-neutral model:
            risk_neutral_scen_obj_value = 1/len(self.data.S_SCENARIOS)*sum(self.model.FirstStageCosts[s] + self.model.SecondStageCosts[s] for s in self.data.S_SCENARIOS)
            return risk_neutral_scen_obj_value

        # give objective function to model
        if risk_neutral:
            self.model.objective_function = Objective(rule=objfun_risk_neutral, sense=minimize) #TEMPORARY: risk-neutral
        else:
            self.model.objective_function = Objective(rule=objfun_risk_averse, sense=minimize) #risk-averse
        
        


        "CONSTRAINTS"

        
        # DEMAND
                
        def FlowRule(model, o, d, p, t,s):
            demand = self.data.D_DEMAND[(o, d, p, t)]
            if  demand < ABSOLUTE_DEVIATION:
                return Constraint.Skip
            else:
                return (sum(self.model.h_path[(k,p,t,s)] for k in self.data.OD_PATHS[(o, d)]) >= demand - FEAS_RELAX)
        self.model.Flow = Constraint(self.data.ODPTS_CONSTR_S, rule=FlowRule)
        # when using uni-modal paths, then this gives an error as there are no paths from World to Hamar/Umeå

        # PATHFLOW

        def PathArcRule(model, i, j, m, r, p, t,s):
            a= (i,j,m,r)
            difference = sum(self.model.x_flow[a, f, p, t,s] for f in self.data.FM_FUEL[m]) - sum(
                self.model.h_path[k, p, t,s] for k in self.data.KA_PATHS[a] )
            return (-ABSOLUTE_DEVIATION,difference,ABSOLUTE_DEVIATION)
        self.model.PathArcRel = Constraint(self.data.APT_CONSTR_S, rule=PathArcRule)

        
        if not self.NoBalancingTrips:
            def PathArcRuleBalancing(model, i, j, m, r, v, t,s):
                a= (i,j,m,r)
                difference = sum(self.model.b_flow[a, f, v, t,s] for f in self.data.FM_FUEL[m]) - sum(
                    self.model.h_path_balancing[k, v, t,s] for k in self.data.KA_PATHS_UNIMODAL[a] )
                return (-ABSOLUTE_DEVIATION,difference,ABSOLUTE_DEVIATION)
            self.model.PathArcRelBalance = Constraint(self.data.AVT_CONSTR_S, rule=PathArcRuleBalancing)

            # FLEET BALANCING
            def FleetBalance1(model, n,m,f,v, t,s):
                disbalance_in_node = (sum(self.model.x_flow[(a, f, p, t,s)] for a in self.data.ANM_ARCS_IN[(n,m)] for p in self.data.PV_PRODUCTS[v]) - 
                        sum(self.model.x_flow[(a, f, p, t, s)] for a in self.data.ANM_ARCS_OUT[(n,m)] for p in self.data.PV_PRODUCTS[v]))  
                empty_trips = (sum(self.model.b_flow[(a, f, v, t,s)] for a in self.data.ANM_ARCS_OUT[(n,m)]) -
                            sum(self.model.b_flow[(a, f, v, t,s)] for a in self.data.ANM_ARCS_IN[(n,m)]))            
                return  (-ABSOLUTE_DEVIATION, disbalance_in_node - empty_trips,ABSOLUTE_DEVIATION)  
            self.model.FleetBalance1 = Constraint(self.data.NMFVT_CONSTR_S, rule=FleetBalance1)



        #-----------------------------------------------#

        # EMISSIONS
        

        def emissions_rule(model, t,s):
            emissions = (
                sum(self.data.E_EMISSIONS[i,j,m,r,f, p, t] * self.model.x_flow[i,j,m,r,f, p, t,s] for p in self.data.P_PRODUCTS
                for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m]) + 
                sum(self.data.E_EMISSIONS[i,j,m,r,f, self.data.cheapest_product_per_vehicle[(m,f,t,v)], t]*EMPTY_VEHICLE_FACTOR * self.model.b_flow[i,j,m,r,f, v, t,s] 
                    for (i,j,m,r) in self.data.A_ARCS for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m])
                - FEAS_RELAX )
            #if t == self.data.T_TIME_PERIODS[0]:
            return (self.model.total_emissions[t,s] == emissions)   # EQUALITY IS NEEDED!
        self.model.Emissions = Constraint(self.data.TS_CONSTR_S, rule=emissions_rule) #removed self.data.T_TIME_PERIODS

        if self.emission_cap_constraint:
            def emission_constr_rule(model,t,s):
                if t > self.data.T_TIME_PERIODS[0]:
                    return self.model.total_emissions[t,s] <= (self.data.EMISSION_CAP_RELATIVE[t]/100*self.model.total_emissions[self.data.T_TIME_PERIODS[0],s] + 
                                                               self.model.emissions_penalty[t,s] + FEAS_RELAX)
                else:
                    return Constraint.Skip
            self.model.EmissionCap = Constraint(self.data.TS_CONSTR_S, rule=emission_constr_rule) #removed self.data.T_TIME_PERIODS

        

        #-----------------------------------------------#
        
        #CAPACITY
        def CapacitatedFlowRule(model,i,j,m,r,ii,jj,mm,rr,t,s):
            e = (i,j,m,r)
            a = (ii,jj,mm,rr)
            return (sum(self.model.x_flow[a, f, p, t, s] for p in self.data.P_PRODUCTS for f in self.data.FM_FUEL[m]) + 
                    sum(self.model.b_flow[a, f, v, t, s] for f in self.data.FM_FUEL[m] for v in self.data.VEHICLE_TYPES_M[m] ) <= 0.5*(self.data.Q_EDGE_BASE[e] +
                + self.data.Q_EDGE_INV[e] * sum(self.model.epsilon_edge[e, tau, s] 
                                                    for tau in self.data.T_TIME_PERIODS 
                                                    if (tau <= (t-self.data.LEAD_TIME_EDGE_INV[e]))    
                                                    and (tau in self.data.T_TIME_FIRST_STAGE)
                                                    )
                                                    )   
                + FEAS_RELAX )
        self.model.CapacitatedFlow = Constraint(self.data.EAT_INV_CONSTR_S, rule = CapacitatedFlowRule)
        
        #Num expansions
        def ExpansionLimitRule(model,i,j,m,r,s):
            e = (i,j,m,r)
            return ( sum(self.model.epsilon_edge[(e,t,s)] for t in self.data.T_TIME_PERIODS if e+(t,) in self.data.ET_INV)<= 1)
        if len(self.data.T_TIME_PERIODS)>1:
            self.model.ExpansionCap = Constraint(self.data.E_EDGES_INV_S, rule = ExpansionLimitRule)


        
        #Terminal capacity constraint. We keep the old notation here, so we can distinguish between OD and transfer, if they take up different capacity.
        def TerminalCapRule(model, i, m,t,s):
            if i in self.data.NM_NODES[m]:
                return (sum(self.model.h_path[k, p, t,s] for k in self.data.ORIGIN_PATHS[(i,m)] for p in self.data.P_PRODUCTS) +
                    sum(self.model.h_path[k, p, t,s] for k in self.data.DESTINATION_PATHS[(i,m)] for p in self.data.P_PRODUCTS) +
                    sum(self.model.h_path[k,p,t,s] for k in self.data.TRANSFER_PATHS[(i,m)] for p in self.data.P_PRODUCTS) <= 
                    self.data.Q_NODE_BASE[i,m]+self.data.Q_NODE_INV[i,m]*sum(self.model.nu_node[i,m,tau,s] 
                                                                                for tau in self.data.T_TIME_PERIODS 
                                                                                if ((tau <= (t-self.data.LEAD_TIME_NODE_INV[i,m])) and
                                                                                    ((i,m) in self.data.NM_CAP_INCR))
                                                                                ) #and((i,m,tau) in self.data.NM_CAP_INCR_T)
                    + FEAS_RELAX )
            else:
                return Constraint.Skip
        self.model.TerminalCap = Constraint(self.data.NM_CAP_T_CONSTR_S, rule = TerminalCapRule)
        

        #Num expansions of terminal ()how many times you can perform a step-wise increase of the capacity)
        def TerminalCapExpRule(model, i, m,s):
            expression = (sum(self.model.nu_node[i,m,t,s] for t in self.data.T_TIME_PERIODS 
                                                        if (t <= self.data.T_TIME_PERIODS[-1] - self.data.LEAD_TIME_NODE_INV[i,m])  ) 
                                                        <= 1)
            if isinstance(expression, bool):
                return Constraint.Skip
            else:
                return(expression) 
        if len(self.data.T_TIME_PERIODS)>1:
            self.model.TerminalCapExp = Constraint(self.data.NM_CAP_INCR_S, rule = TerminalCapExpRule)

        

        #Charging / Filling
        def ChargingCapArcRule(model, i, j, m, r,f, t,s):
            e = (i, j, m, r)
            return (sum(self.model.x_flow[a,f,p, t,s] for p in self.data.P_PRODUCTS
                    for a in self.data.AE_ARCS[e]) + sum(self.model.b_flow[a,f,v, t,s] for a in self.data.AE_ARCS[e] 
                        for v in self.data.VEHICLE_TYPES_M[m]) <= (self.data.Q_CHARGE_BASE[(e,f)] + self.model.y_charge[(e,f,t,s)] )  
                                                                    + FEAS_RELAX )
        self.model.ChargingCapArc = Constraint(self.data.EFT_CHARGE_CONSTR_S, rule=ChargingCapArcRule)
        #AIM also looked into charging infrastructure in NODES

        #Upgrading
        def InvestmentInfraRule(model,i,j,m,r,f,t,s):
            e = (i,j,m,r)
            return (sum(self.model.x_flow[a,f,p,t,s] for p in self.data.P_PRODUCTS for a in self.data.AE_ARCS[e])
                    <= self.data.BIG_M_UPG[e]*sum(self.model.upsilon_upg[e,f,tau,s] 
                                                for tau in self.data.T_TIME_PERIODS 
                                                if (tau <= (t-self.data.LEAD_TIME_EDGE_UPG[(e,f)])) and
                                                    (tau in self.data.T_TIME_FIRST_STAGE)   #and((e,f,tau) in self.data.UT_UPG)
                                                )   
                    + FEAS_RELAX )
        self.model.InvestmentInfra = Constraint(self.data.UT_UPG_CONSTR_S, rule = InvestmentInfraRule)
        
        

        #-----------------------------------------------#
    
        #TransportArbeid
        def TotalTransportAmountRule(model,m,f,t,s):
            return (self.model.q_transp_amount[m,f,t,s] == sum( self.data.AVG_DISTANCE[a]*self.model.x_flow[a,f,p,t,s] for p in self.data.P_PRODUCTS 
                                                            for a in self.data.AM_ARCS[m]))
        self.model.TotalTranspAmount = Constraint(self.data.MFT_CONSTR_S, rule = TotalTransportAmountRule)

        #Technology maturity limit upper bound (which also comes from bass diffusion)
        def TechMaturityLimitRule(model, m, f, t,s):
            return (self.model.q_transp_amount[(m,f,t,s)] <= round(self.data.R_TECH_READINESS_MATURITY[(m,f,t,s)]/100,NUM_DIGITS_PRECISION)*sum(self.model.q_transp_amount[(m,ff,t,s)] for ff in self.data.FM_FUEL[m])
            + FEAS_RELAX )   #R_TECH is the A from the Bass diffusion model
        self.model.TechMaturityLimit = Constraint(self.data.MFT_MATURITY_CONSTR_S, rule = TechMaturityLimitRule)

        #HFO is toxic and will be phased out -> put a cap
        # def PhaseOutRule(model, m, f, t,s):
        #     if ((m,f,t) not in self.data.PHASE_OUT.keys()):
        #         return Constraint.Skip  
        #     else:
        #         if self.single_time_period:
        #             return (self.model.q_transp_amount[(m,f,t,s)] <= self.data.PHASE_OUT[(m,f,t)]*sum(self.model.q_transp_amount[(m,ff,t,s)] for ff in self.data.FM_FUEL[m]))
        #         else:
        #             return (self.model.q_transp_amount[(m,f,t,s)] <= (self.data.PHASE_OUT[(m,f,t)]*self.model.q_transp_amount[(m,f,self.data.T_MIN1[t],s)]))
        # self.model.PhaseOut = Constraint(self.data.MFT_MIN0_S, rule = PhaseOutRule)
        def PhaseOutRule(model, m, f, t,s):
            if ((m,f,t) not in self.data.PHASE_OUT.keys()):
                return Constraint.Skip  
            else:
                if t >= self.data.T_TIME_PERIODS[2]:  #2034
                    return (self.model.q_transp_amount[(m,f,t,s)] <= 0)
                else:
                    return Constraint.Skip
        self.model.PhaseOut = Constraint(self.data.MFT_MIN0_S, rule = PhaseOutRule)


        if True:


            if self.single_time_period is None:
                
                
                if True:
                    #Initialize the transport amounts (put an upper bound at first)
                    def InitTranspAmountRule(model, m, f, t,s):
                        return (self.model.q_transp_amount[(m,f,t,s)] <= round(self.data.Q_SHARE_INIT_MAX[(m,f,t)]/100,NUM_DIGITS_PRECISION)*sum(self.model.q_transp_amount[(m,ff,t,s)] for ff in self.data.FM_FUEL[m]))   
                    self.model.InitTranspAmount = Constraint(self.data.MFT_INIT_TRANSP_SHARE_S, rule = InitTranspAmountRule)
                    
                    #Note, due to our initialization, this one can lead to infeasibility due to capacity limits on rail.  
                    def InitialModeSplit(model,m,s):
                        if m=='Rail':
                            return Constraint.Skip                    
                        else:
                            t0 = self.data.T_TIME_PERIODS[0]
                            total_transport_amount = sum(self.model.q_transp_amount[(mm,f,t0,s)] for mm in self.data.M_MODES for f in self.data.FM_FUEL[mm])
                            modal_transport_amount = sum(self.model.q_transp_amount[(m,f,t0,s)] for f in self.data.FM_FUEL[m])
                            return modal_transport_amount >= (self.data.INIT_MODE_SPLIT[m]-0.01)*total_transport_amount    
                    self.model.InitialModeSplitConstr = Constraint(self.data.M_MODES_S,rule = InitialModeSplit)
            
                
                    #Auxiliary transport amount (q_aux equal to q in all decision periods t)
                    def AuxTransportAmountRule(model,m,f,t,s):
                        return (self.model.q_aux_transp_amount[m,f,t,s] == self.model.q_transp_amount[m,f,t,s]) #auxiliary q variable equal to "normal" q variable 
                    self.model.AuxTranspAmount = Constraint(self.data.MFT_NEW_S, rule = AuxTransportAmountRule) #note: only in decision periods

                    #Total mode transport amount (q_mode_total equal to sum over q_aux for all f in F[m], for decision periods)
                    def ModeTotalTransportAmountRule(model,m,t,s):
                        return (self.model.q_mode_total_transp_amount[m,t,s] == sum( self.model.q_transp_amount[m,f,t,s] for f in self.data.FM_FUEL[m] ))
                    self.model.ModeTotalTransportAmount = Constraint(self.data.MT_S, rule = ModeTotalTransportAmountRule) 

                    #Restriction on modal transport amount decrease (mode shift cannot happen too fast)
                    def DecreaseInModeTotalTransportAmountRule(model,m,t,s):
                        return (self.model.q_mode_total_transp_amount[m,t,s] >= RHO_STAR**(t-self.data.T_MIN1[t])*self.model.q_mode_total_transp_amount[m,self.data.T_MIN1[t],s])
                    self.model.DecreaseModeTotalTransportAmount = Constraint(self.data.MT_MIN0_S, rule = DecreaseInModeTotalTransportAmountRule)
                    #Restriction on modal transport amount increase (mode shift cannot happen too fast)
                    def IncreaseInModeTotalTransportAmountRule(model,m,t,s):
                        return (self.model.q_mode_total_transp_amount[m,t,s] <= (2-RHO_STAR)**(t-self.data.T_MIN1[t])*self.model.q_mode_total_transp_amount[m,self.data.T_MIN1[t],s])
                    self.model.IncreaseModeTotalTransportAmount = Constraint(self.data.MT_MIN0_S, rule = IncreaseInModeTotalTransportAmountRule)

                    

                if True:        
                    #Fleet Renewal
                    def FleetRenewalRule(model,m,t,s):
                        all_decreases = sum(self.model.q_transp_delta[(m,f,t,s)] for f in self.data.FM_FUEL[m])
                        factor = (t - self.data.T_MIN1[t]) / self.data.LIFETIME[m]
                        return (all_decreases <= factor*self.model.q_mode_total_transp_amount[m,self.data.T_MIN1[t],s])
                    self.model.FleetRenewal = Constraint(self.data.MT_MIN0_S, rule = FleetRenewalRule)
                    
                    def FleetRenewalPosPartRule(model,m,f,t,s):
                        decrease = self.model.q_transp_amount[(m,f,self.data.T_MIN1[t],s)] - self.model.q_transp_amount[(m,f,t,s)]
                        return (self.model.q_transp_delta[(m,f,t,s)] >= decrease)
                    self.model.FleetRenewalPosPart = Constraint(self.data.MFT_MIN0_S, rule = FleetRenewalPosPartRule)

                if True:
                    #-- Bass diffusion ----------------------

                    #most likely not needed
                    #Bass diffusion paths first period (initial values): q[t] <= (2022 - t_0)^+ * alpha * q_bar[t]
                    def BassDiffusionRuleFirstPeriod(model,m,f,t,s):
                        pos_part = max(self.data.T_TIME_PERIODS[0] - self.data.tech_base_bass_model[(m,f)].t_0, 0)
                        return ( self.model.q_transp_amount[m,f,t,s] <= pos_part * self.data.tech_base_bass_model[(m,f)].p * self.model.q_mode_total_transp_amount[m,t,s] )
                    self.model.BassDiffusionFirstPeriod = Constraint(self.data.MFT_NEW_FIRST_PERIOD_S, rule = BassDiffusionRuleFirstPeriod)

                    # only add remaining Bass diffusion constraints if we run the full model (not just first period in the initialization run)
                    if len(self.data.T_TIME_PERIODS) > 1:
                        #Bass diffusion paths (1st stage): change in q is at most alpha * q_bar[t-1] + beta * q[t-1]    (based on pessimistic beta)
                        def BassDiffusionRuleFirstStage(model,m,f,t,s):
                            diff_has_started = int(t >= self.data.tech_base_bass_model[(m,f)].t_0) # boolean indicating whether diffusion process has started at time t
                            return ( self.model.q_aux_transp_amount[m,f,t,s] - self.model.q_aux_transp_amount[m,f,t-1,s] 
                                <= diff_has_started * (self.data.tech_base_bass_model[(m,f)].p * self.model.q_mode_total_transp_amount[m,self.data.T_MOST_RECENT_DECISION_PERIOD[t-1],s]
                                #+ (1 - self.data.tech_scen_p_q_variation[(m,f)]) * self.data.tech_active_bass_model[(m,f,s)].q * self.model.q_aux_transp_amount[m,f,t-1] ))   #initial pessimistic path
                                + self.data.tech_base_bass_model[(m,f)].q * self.model.q_aux_transp_amount[m,f,t-1,s] ))   #initial base path
                        self.model.BassDiffusionFirstStage = Constraint(self.data.MFT_NEW_YEARLY_FIRST_STAGE_MIN0_S, rule = BassDiffusionRuleFirstStage)

                        # Bass diffusion paths (2nd stage): change in q is at most alpha * q_bar[t-1] + beta * q[t-1]   (based on scenario beta)
                        def BassDiffusionRuleSecondStage(model,m,f,t,s):
                            diff_has_started = int(t >= self.data.tech_base_bass_model[(m,f)].t_0) # boolean indicating whether diffusion process has started at time t
                            return ( self.model.q_aux_transp_amount[m,f,t,s] - self.model.q_aux_transp_amount[m,f,t-1,s] 
                                <= diff_has_started * ( self.data.tech_active_bass_model[(m,f,s)].p * self.model.q_mode_total_transp_amount[m,self.data.T_MOST_RECENT_DECISION_PERIOD[t-1],s] 
                                + self.data.tech_active_bass_model[(m,f,s)].q * self.model.q_aux_transp_amount[m,f,t-1,s] ) )  
                                # q(t) - q(t-1) <= alpha * q_bar(|_t-1_|) + beta * q(t-1)
                        self.model.BassDiffusionSecondStage = Constraint(self.data.MFT_NEW_YEARLY_SECOND_STAGE_S, rule = BassDiffusionRuleSecondStage)

        #-----------------------------------------------#

        # NON ANTICIPATIVIY

        def combinations(list_of_tuples1, list_of_tuples2):
            if type(list_of_tuples1[0]) is not tuple:
                list_of_tuples1 = [(x,) for x in list_of_tuples1]
            if type(list_of_tuples2[0]) is not tuple:
                list_of_tuples2 = [(x,) for x in list_of_tuples2]
            output = [x+y for x in list_of_tuples1 for y in list_of_tuples2]
            return output

        if len(self.data.S_SCENARIOS)>1:
            
            ABSOLUTE_DEVIATION_NONANT = 0 #not smart to relax this

            def Nonanticipativity_x(model,i,j,m,r,f,p,t,s,ss):
                a = i,j,m,r
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): # Gurobi removes when already added
                    diff = (self.model.x_flow[(a,f,p,t,s)]- self.model.x_flow[(a,f,p,t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip   
            self.model.Nonanticipativity_x_Constr = Constraint(combinations(self.data.AFPT,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_x)

            def Nonanticipativity_b(model,i,j,m,r,f,v,t,s,ss):
                a = i,j,m,r
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = self.model.b_flow[(a,f,v,t,s)]- self.model.b_flow[(a,f,v,t,ss)]
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_b_Constr = Constraint(combinations(self.data.AFVT,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_b)

            def Nonanticipativity_h(model,k,p,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.h_path[(k,p,t,s)]- self.model.h_path[(k,p,t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_h_Constr = Constraint(combinations(self.data.KPT,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_h)

            def Nonanticipativity_h_bal(model,k,v,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.h_path_balancing[(k,v,t,s)]- self.model.h_path_balancing[(k,v,t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_h_bal_Constr = Constraint(combinations(self.data.KVT,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_h_bal)

            def Nonanticipativity_total_emissions(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.total_emissions[(t,s)]- self.model.total_emissions[(t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT) 
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_total_emissions_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_total_emissions)


            def Nonanticipativity_stage(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.StageCosts[(t,s)]- self.model.StageCosts[(t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_stage_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_stage)

            def Nonanticipativity_eps(model,i,j,m,r,t,s,ss):
                e = (i,j,m,r)
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.epsilon_edge[(e,t,s)]- self.model.epsilon_edge[(e,t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            if len(self.data.ET_INV)>0:
                self.model.Nonanticipativity_eps_Constr = Constraint(combinations(self.data.ET_INV,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_eps)

            def Nonanticipativity_upg(model,i,j,m,r,f,t,s,ss):
                e = (i,j,m,r)
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.upsilon_upg[(e,f,t,s)]- self.model.upsilon_upg[(e,f,t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            if len(self.data.UT_UPG)>0:
                self.model.Nonanticipativity_upg_Constr = Constraint(combinations(self.data.UT_UPG,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_upg)

            def Nonanticipativity_nu(model,n,m,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.nu_node[(n,m,t,s)]- self.model.nu_node[(n,m,t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            if len(self.data.NM_CAP_INCR_T)>0:    
                self.model.Nonanticipativity_nu_Constr = Constraint(combinations(self.data.NM_CAP_INCR_T,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_nu)

            def Nonanticipativity_y(model,i,j,m,r,f,t,s,ss):
                e = (i,j,m,r)
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.y_charge[(e,f,t,s)]- self.model.y_charge[(e,f,t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            if len(self.data.EFT_CHARGE)>0: 
                self.model.Nonanticipativity_y_Constr = Constraint(combinations(self.data.EFT_CHARGE,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_y)

            def Nonanticipativity_q(model,m,f,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.q_transp_amount[(m,f,t,s)]- self.model.q_transp_amount[(m,f,t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            if len(self.data.MFT)>0: 
                self.model.Nonanticipativity_q_Constr = Constraint(combinations(self.data.MFT,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_q)

            def Nonanticipativity_q_delta(model,m,f,t,s,ss): 
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.q_transp_delta[(m,f,t,s)]- self.model.q_transp_delta[(m,f,t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip    
            if len(self.data.MFT_MIN0)>0:
                self.model.Nonanticipativity_qdelta_Constr = Constraint(combinations(self.data.MFT_MIN0,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_q_delta)

            def Nonanticipativity_q_aux(model,m,f,t,s,ss):
                if (t in self.data.T_YEARLY_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.q_aux_transp_amount[(m,f,t,s)]- self.model.q_aux_transp_amount[(m,f,t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_qaux_Constr = Constraint(combinations(self.data.MFT_NEW_YEARLY,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_q_aux)
            
            def Nonanticipativity_qmode(model,m,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff= (self.model.q_mode_total_transp_amount[(m,t,s)]- self.model.q_mode_total_transp_amount[(m,t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_qmode_Constr = Constraint(combinations(self.data.MT,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_qmode)

            def Nonanticipativity_opex(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.TranspOpexCost[(t,s)]- self.model.TranspOpexCost[(t,ss)])
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT) 
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_opex_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_opex)

            def Nonanticipativity_co2(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.TranspCO2Cost[(t,s)]- self.model.TranspCO2Cost[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_co2_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_co2)

            def Nonanticipativity_co2_pen(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.CO2_PENALTY[(t,s)]- self.model.CO2_PENALTY[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_co2_pen_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_co2_pen)

            def Nonanticipativity_opexb(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.TranspOpexCostB[(t,s)]- self.model.TranspOpexCostB[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_opexb_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_opexb)

            def Nonanticipativity_co2b(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.TranspCO2CostB[(t,s)]- self.model.TranspCO2CostB[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_co2b_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_co2b)

            def Nonanticipativity_transf(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.TransfCost[(t,s)]- self.model.TransfCost[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_transf_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_transf)

            def Nonanticipativity_edge(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.EdgeCost[(t,s)]- self.model.EdgeCost[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_edge_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_edge)

            def Nonanticipativity_node(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.NodeCost[(t,s)]- self.model.NodeCost[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_node_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_node)

            def Nonanticipativity_upgcost(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.UpgCost[(t,s)]- self.model.UpgCost[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_upgcost_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_upgcost)

            def Nonanticipativity_chargecost(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.ChargeCost[(t,s)]- self.model.ChargeCost[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_chargecost_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_chargecost)

            def Nonanticipativity_fillingcost(model,t,s,ss):
                if (t in self.data.T_TIME_FIRST_STAGE) and (s is not ss): 
                    diff = (self.model.FillingCost[(t,s)]- self.model.FillingCost[(t,ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_fillingcost_Constr = Constraint(combinations(self.data.T_TIME_PERIODS,self.data.SS_SCENARIOS_NONANT),rule = Nonanticipativity_fillingcost)


            def Nonanticipativity_firststagecost(model,s,ss):
                if  (s is not ss): 
                    diff = (self.model.FirstStageCosts[(s)]- self.model.FirstStageCosts[(ss)]) 
                    return (-ABSOLUTE_DEVIATION_NONANT,diff,ABSOLUTE_DEVIATION_NONANT)
                else:
                    return Constraint.Skip
            self.model.Nonanticipativity_firststagecost_Constr = Constraint(self.data.SS_SCENARIOS_NONANT,rule = Nonanticipativity_firststagecost)

            # self.model.CvarAux   -> non-anticipativity already imposed in the model (not defined for scenarios)    
            # self.model.CvarPosPart -> Different across scenarios
            #self.model.SecondStageCosts  -> varies across scenarios


        #-----------------------------------------------#
        if NO_INVESTMENTS:
            for (i,j,m,r,tau,s) in self.data.ET_IN_S:
                self.model.epsilon_edge[i,j,m,r, tau,s].fix(0)
            for (e,f,t,s) in self.data.UT_UPG_S:
                self.model.upsilon_upg[(e,f,t,s )].fix(0)
            for (n,m,t,s ) in self.data.NM_CAP_INCR_T_S:
                self.model.nu_node[((n,m,t,s ))].fix(0)

        return self.model
    
    
    def fix_first_time_period(self, model_instance_init):
    
        #When fixing the first_stage flow, we quickly run into feasibility issues. So, let's only fix the emissions.
        if self.single_time_period is None:
            for s in self.data.S_SCENARIOS:
                t = self.data.T_TIME_PERIODS[0]
                weight = model_instance_init.model.total_emissions[(t,s)].value
                self.model.total_emissions[(t,s)].setub((1+RELATIVE_DEVIATION)*weight)
                self.model.total_emissions[(t,s)].setlb((1-RELATIVE_DEVIATION)*weight)
                
            # t = self.data.T_TIME_PERIODS[0]
            # for scen_name in self.data.S_SCENARIOS:
            #     for (i,j,m,r) in self.data.A_ARCS:
            #         a = (i,j,m,r)
            #         for f in self.data.FM_FUEL[m]:
            #             for p in self.data.P_PRODUCTS:
            #                 weight = model_instance_init.model.x_flow[(a,f,p,t,scen_name)].value
            #                 if weight is not None: 
            #                     if weight != 0: 
            #                         self.model.x_flow[(a,f,p,t,scen_name)].setub((1+RELATIVE_DEVIATION)*weight)
            #                         self.model.x_flow[(a,f,p,t,scen_name)].setlb((1-RELATIVE_DEVIATION)*weight)
            #                         #self.model.x_flow[(a,f,p,t,s)].fix(weight) 
            #                     else:
            #                         #self.model.x_flow[(a,f,p,t,s)].fix(0)  #infeasibilities
            #                         self.model.x_flow[(a,f,p,t,scen_name)].setlb(-ABSOLUTE_DEVIATION)
            #                         self.model.x_flow[(a,f,p,t,scen_name)].setub(ABSOLUTE_DEVIATION)

            # for (i,j,m,r,f,p,t,scen_name) in self.data.AFVT_S:   #This one leads to infeasibility unfortunately, cannot fix too much.
            #     a = (i,j,m,r)
            #     if t== self.data.T_TIME_PERIODS[0]:     
            #         weight = model_instance_init.model.b_flow[(a,f,p,t,scen_name)].value
            #         if weight is not None: 
            #             if weight != 0: 
            #                 self.model.b_flow[(a,f,p,t,scen_name)].setub((1+RELATIVE_DEVIATION)*weight)
            #                 self.model.b_flow[(a,f,p,t,scen_name)].setlb((1-RELATIVE_DEVIATION)*weight)
            #                 #self.model.x_flow[(a,f,p,t,s)].fix(weight) 
            #             else:
            #                 #self.model.x_flow[(a,f,p,t,s)].fix(0)  #infeasibilities
            #                 self.model.b_flow[(a,f,p,t,scen_name)].setlb(-ABSOLUTE_DEVIATION)
            #                 self.model.b_flow[(a,f,p,t,scen_name)].setub(ABSOLUTE_DEVIATION)
        
                    

    
    def fix_variables_first_stage(self,model_ev):  #this is the EV model that is input.
        
        
        # for v in solved_init_model.model.component_objects(pyo.Var, active=True):
        #     #print("Variable",v)  
        #     for index in v:
        #         if v[index].value is not None:
        #             #print ("   ",index, pyo.value(v[index]))
        #             var_big_model = getattr(self.model,str(v))
        #             var_big_model[index].fix(v[index].value)  


        # for (i,j,m,r,f,p,t) in self.data.AFPT_S:    
        #     a = (i,j,m,r)
        #     if t in self.data.T_TIME_FIRST_STAGE:
        #         self.model.x_flow[(a,f,p,t)].fix(0)
        
        base_scenario = 'BB'  # "BBB" OR "BB", update to pick the right one automatically

        
        for s in self.data.S_SCENARIOS:
            for t in self.data.T_TIME_FIRST_STAGE:
                weight = model_ev.total_emissions[(t,base_scenario)].value
                self.model.total_emissions[(t,s)].setub((1+RELATIVE_DEVIATION)*weight)
                self.model.total_emissions[(t,s)].setlb((1-RELATIVE_DEVIATION)*weight)
        
        # for (i,j,m,r,f,p,t,s) in self.data.AFPT_S:
        #     a = (i,j,m,r)
        #     self.model.x_flow[(a,f,p,t,s)].fixed = False
        #     if t in self.data.T_TIME_FIRST_STAGE:
        #         weight = model_ev.x_flow[(a,f,p,t,base_scenario)].value
        #         if weight is not None:
        #             if weight != 0: 
        #                 self.model.x_flow[(a,f,p,t,s)].setub((1+RELATIVE_DEVIATION)*weight)
        #                 self.model.x_flow[(a,f,p,t,s)].setlb((1-RELATIVE_DEVIATION)*weight)
        #                 #self.model.x_flow[(a,f,p,t,s)].fix(weight) 
        #             else:
        #                 #self.model.x_flow[(a,f,p,t,s)].fix(0)  #infeasibilities
        #                 self.model.x_flow[(a,f,p,t,s)].setlb(-ABSOLUTE_DEVIATION)
        #                 self.model.x_flow[(a,f,p,t,s)].setub(ABSOLUTE_DEVIATION)
        #         else:
        #             pass #this does not happen

        # for (i,j,m,r,f,v,t,s) in self.data.AFVT_S:
        #     a = (i,j,m,r)
        #     self.model.b_flow[(a,f,v,t,s)].fixed = False
        #     if t in self.data.T_TIME_FIRST_STAGE:
        #         weight = model_ev.b_flow[(a,f,v,t,base_scenario)].value
        #         if weight is not None:
        #             if weight != 0: 
        #                 self.model.b_flow[(a,f,v,t,s)].setub((1+RELATIVE_DEVIATION)*weight)
        #                 self.model.b_flow[(a,f,v,t,s)].setlb((1-RELATIVE_DEVIATION)*weight)
        #                 #self.model.b_flow[(a,f,v,t,s)].fix(weight) 
        #             else:
        #                 #self.model.b_flow[(a,f,p,t,s)].fix(0)  #infeasibilities
        #                 self.model.b_flow[(a,f,v,t,s)].setlb(-ABSOLUTE_DEVIATION)
        #                 self.model.b_flow[(a,f,v,t,s)].setub(ABSOLUTE_DEVIATION)
        #         else:
        #             pass #this does not happen

        for (i,j,m,r,t,s) in self.data.ET_INV_S:
            if t in self.data.T_TIME_FIRST_STAGE:
                e = (i,j,m,r)
                weight = model_ev.epsilon_edge[(e,t,base_scenario)].value
                if weight is not None:
                    if weight > 0:
                        self.model.epsilon_edge[(e,t,s)].fix(weight) 
                    else:
                        self.model.epsilon_edge[(e,t,s)].fix(0) 
        for (e,f,t,s) in self.data.UT_UPG_S:
            if t in self.data.T_TIME_FIRST_STAGE:
                weight = model_ev.upsilon_upg[(e,f,t,base_scenario)].value
                if weight is not None:
                    if weight > 0:
                        self.model.upsilon_upg[(e,f,t,s)].fix(weight) 
                    else:
                        self.model.upsilon_upg[(e,f,t,s)].fix(0)

        for (i,m,t,s) in self.data.NM_CAP_INCR_T_S:
            if t in self.data.T_TIME_FIRST_STAGE:
                weight = model_ev.nu_node[(i,m,t,base_scenario)].value
                if weight is not None:
                    if weight > 0:
                        self.model.nu_node[(i,m,t,s)].fix(weight)
                    else:
                        self.model.nu_node[(i,m,t,s)].fix(0)

        for (e,f,t,s) in self.data.EFT_CHARGE_S:
            if t in self.data.T_TIME_FIRST_STAGE:
                weight = model_ev.y_charge[(e,f,t,base_scenario)].value
                if weight is not None:
                    if weight > 0:
                        self.model.y_charge[(e,f,t,s)].fix(weight)
                    else:
                        self.model.y_charge[(e,f,t,s)].fix(0)
        
        
    def unfix_variables_first_stage(self):
        
        #relax=0.001  #5*10**(-2)
        #self.model.feas_relax.setlb(relax)
        #self.model.feas_relax.setub(relax)
        
        upperbound = max(self.data.D_DEMAND.values())*2
        for (i,j,m,r,f,p,t,s) in self.data.AFPT_S:
            a = (i,j,m,r)
            self.model.x_flow[(a,f,p,t,s)].fixed = False
            if t in self.data.T_TIME_FIRST_STAGE:
                if t is not self.data.T_TIME_FIRST_STAGE[0]: #still keep the first one fixed
                    self.model.x_flow[(a,f,p,t,s)].fixed = False
                    self.model.x_flow[(a,f,p,t,s)].setlb(-ABSOLUTE_DEVIATION)
                    self.model.x_flow[(a,f,p,t,s)].setub(upperbound)   #maximum value is below 1000

        for (i,j,m,r,f,v,t,s) in self.data.AFVT_S:
            a = (i,j,m,r)
            self.model.b_flow[(a,f,v,t,s)].fixed = False
            if t in self.data.T_TIME_FIRST_STAGE:
                if t is not self.data.T_TIME_FIRST_STAGE[0]: #still keep the first one fixed!!
                    self.model.b_flow[(a,f,v,t,s)].fixed = False
                    self.model.b_flow[(a,f,v,t,s)].setlb(-ABSOLUTE_DEVIATION)
                    self.model.b_flow[(a,f,v,t,s)].setub(upperbound)   #maximum value is below 1000

        for (i,j,m,r,t,s) in self.data.ET_INV_S:
            e = (i,j,m,r)
            self.model.epsilon_edge[(e,t,s)].fixed = False
        
        for (e,f,t,s) in self.data.UT_UPG_S:
            self.model.upsilon_upg[(e,f,t,s)].fixed = False

        for (i,m,t,s) in self.data.NM_CAP_INCR_T_S:
            self.model.nu_node[(i,m,t,s)].fixed = False

        for (e,f,t,s) in self.data.EFT_CHARGE_S:
            self.model.y_charge[(e,f,t,s)].fixed = False
                    
    
            

    def solve_model(self, warmstart=False, 
                    FeasTol=(10**(-2)), 
                    MIP_gap=MIPGAP,
                    # 'TimeLimit':600, # (seconds)
                    num_focus= 0,  # 0 is automatic, 1 is low precision but fast  https://www.gurobi.com/documentation/9.5/refman/numericfocus.html
                    Crossover=-1, #default: -1, automatic, https://www.gurobi.com/documentation/9.1/refman/crossover.html
                    Method=-1, #root node relaxation, def: -1    https://www.gurobi.com/documentation/9.1/refman/method.html
                    NodeMethod=-1):  # all other nodes, https://www.gurobi.com/documentation/9.1/refman/nodemethod.html

        opt = pyomo.opt.SolverFactory('gurobi') #gurobi
        opt.options['FeasibilityTol'] = FeasTol 
        opt.options['MIPGap']= MIP_gap 
        opt.options["NumericFocus"] = num_focus
        opt.options["BarConvTol"] = 1E-8 # default: 1E-8https://www.gurobi.com/documentation/9.1/refman/barconvtol.html
        opt.options["Crossover"] = Crossover
        opt.options["Method"] = Method 
        opt.options["NodeMethod"] = NodeMethod 
        #opt.options["DualReductions"] = 0 #default 1. At zero, figure out if unbounded or infeasible.
        results = opt.solve(self.model, warmstart=warmstart, tee=True, 
                                        symbolic_solver_labels=False, #goes faster, but turn to true with errors!
                                        keepfiles=False)  
                                        #https://pyomo.readthedocs.io/en/stable/working_abstractmodels/pyomo_command.html
        if True:
            solver_stat = results.solver.status
            termination_cond = results.solver.termination_condition
            if (solver_stat == pyomo.opt.SolverStatus.ok) and (
                    termination_cond == pyomo.opt.TerminationCondition.optimal):
                print('the solution is feasible and optimal')
            else:
                log_infeasible_constraints(self.model)
                log_infeasible_constraints(self.model, log_expression=True, log_variables=True)
                logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)
                raise Exception('Solver Status: ', solver_stat, 'Termination Condition: ', termination_cond )
                #print('Solver Status: '), self.results.solver.status
                #print('Termination Condition: '), self.results.solver.termination_condition

            print('Solution time: ' + str(results.solver.time))
        
        #self.model.EmissionCap.pprint()
        #self.model.total_emissions.display() #print()

        
