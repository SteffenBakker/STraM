#Pyomo
import pyomo.opt   # we need SolverFactory,SolverStatus,TerminationCondition
import pyomo.opt.base as mobase
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints
import logging
from Data.settings import *
from Data.Create_Sets_Class import TransportSets   #FreightTransportModel.Data.Create_Sets_Class
import os
os.getcwd()

############# Class ################



class TranspModel:

    def __init__(self, instance, base_data, one_time_period, scenario, carbon_scenario, fuel_costs, emission_reduction):

        self.instance = instance
        #timelimit in minutes, etc
        #elf.maturity_scenario = maturity_scenario

        self.results = 0  # results is an structure filled out later in solve_model()
        self.status = ""  # status is a string filled out later in solve_model()
        self.model = ConcreteModel()
        self.opt = pyomo.opt.SolverFactory('gurobi') #gurobi
        self.results = 0  # results is an structure filled out later in solve_model()
        self.status = ""  # status is a string filled out later in solve_model()
        self.scenario = scenario
        self.carbon_scenario = carbon_scenario
        self.fuel_costs = fuel_costs
        self.emission_reduction = emission_reduction
        #IMPORT THE DATA
        self.data = base_data
        self.data.update_parameters(scenario, carbon_scenario, fuel_costs, emission_reduction)
        self.factor = self.data.factor



    def construct_model(self):
        
        #Significant speed improvements can be obtained using the LinearExpression object when there are long, dense, linear expressions.
        # USe linearexpressions: https://pyomo.readthedocs.io/en/stable/advanced_topics/linearexpression.html
        

        "VARIABLES"
        # Binary, NonNegativeReals, PositiveReals, etc

        #self.model.x_flow_rest = Var(self.data.APTS_rest, within=NonNegativeReals)
        self.model.x_flow = Var(self.data.AFPT, within=NonNegativeReals)
        self.model.h_flow = Var(self.data.KPT, within=NonNegativeReals)# flow on paths K,p
        #self.model.h_flow_2020 = Var(self.data.KPTS_2020, within=NonNegativeReals)
        self.model.StageCosts = Var(self.data.T_TIME_PERIODS, within = NonNegativeReals)
        
        self.model.v_edge = Var(self.data.ET_RAIL, within = Binary) #within = Binary
        self.model.u_upg = Var(self.data.EFT_UPG, within = Binary) #bin.variable for investments upgrade/new infrastructure u at link l, time period t
        self.model.w_node = Var(self.data.NCMT, within = Binary) #step-wise investment in terminals
        
        #self.model.y_exp_link = Var(self.data.ET_RAIL, within = NonNegativeReals) #,bounds=(0,20000)) #expansion in tonnes for capacitated links l, time period t
        #self.model.y_exp_node = Var(self.data.NMT_CAP, within = NonNegativeReals, bounds=(0,100000000)) #expansion in tonnes for capacicated terminals i, time period t
        
        self.model.y_charge = Var(self.data.EFT_CHARGE, within=NonNegativeReals)
        #self.model.charge_node = Var(self.data.CHARGING_IMFT, within=NonNegativeReals)
        self.model.z_emission = Var(self.data.TS, within = NonNegativeReals)
        
        
        """def emission_bound(model, t):
            return (0, self.data.CO2_CAP[t]/self.factor)"""

        self.model.total_emissions = Var(self.data.TS, within=NonNegativeReals) #instead of T_PERIODS!
                                        # bounds=emission_bound)  # just a variable to help with output
        
        #TO DO: check how the CO2 kicks in? Why is it there? Should be baked into C_TRANSP_COST
        def StageCostsVar(model, t):
            return(self.model.StageCosts[t] == (
                sum(self.data.D_DISCOUNT_RATE**n*(self.data.C_TRANSP_COST[(a,f,p,t)]+self.data.C_CO2[(a,f,p,t)])*
                    self.model.x_flow[(a,f,p,t)]
                for a in self.data.A_ARCS for f in self.data.F_FUELS_ARC for n in self.data.Y_YEARS[t]) +
                sum(self.data.D_DISCOUNT_RATE^n*self.data.C_TRANSFER[(k,p)] 
                for k in self.MULTI_MODE_PATHS for p in self.P_PRODUCTS for n in self.data.Y_YEARS[t]) + 
                self.data.D_DISCOUNT_RATE**self.data.Y_YEARS[t][0](
                    sum(self.data.C_EDGE_RAIL[e]*self.model.v_edge[(e,t)] for e in self.E_EDGES_RAIL) +
                    sum(self.data.C_NODE[(i,c,m)]*self.model.w_node[(i,c,m,t)] for (i, m) in self.NM_LIST_CAP for c in self.TERMINAL_TYPE[m] ) +
                    # maybe N_NODES_CAP_NORWAY is needed?
                    sum(self.data.C_UPG[(e,f)]*self.model.u_upg[(e,f,t)] for (e,f) in self.data.U_UPGRADE) +
                    sum(self.data.C_CHARGE[(e,f)])*self.model.y_charge[(e,f,t)] (e,f) in self.CHARGING_EDGES_FUELS)
                    ) +
                EMISSION_VIOLATION_PENALTY*self.model.z_emission[t]
                )
                
        self.model.stage_costs = Constraint(self.data.T_TIME_PERIODS, rule = StageCostsVar)
        
        
        def objfun(model):
            #obj_value = self.model.first_stage_costs + sum(self.data.D_DISCOUNT_RATE[t] * self.data.C_TRANSP_COST[a, t]*self.model.x_flow[a,p,t]
             #                for p in self.data.P_PRODUCTS for a in self.data.A_ARCS for t
              #               in [2025,2030])
            obj_value = sum(self.model.StageCosts[t] for t in self.data.T_TIME_PERIODS) 
            return obj_value

        self.model.Obj = Objective(rule=objfun, sense=minimize)

        # Flow
        def FlowRule(model, o, d, p, t):
            return sum(self.model.h_flow[str(k), p, t] for k in self.data.OD_PATHS[(o, d)]) >= self.data.D_DEMAND[
                (o, d, p, t)]/self.factor

        # NOTE THAT THIS SHOULD BE AN EQUALITY; BUT THEN THE PROBLEM GETS EASIER WITH A LARGER THAN OR EQUAL
        self.model.Flow = Constraint(self.data.ODPTS, rule=FlowRule)
        #print("OVER FEILEN!")

        # PathFlow-ArcFlow relationship
       
        def PathArcRule(model, i, j, m, r, p, t):
            l= (i,j,m,r)
            return sum(self.model.x_flow[a, p, t] for a in self.data.A_LINKS[l]) == sum(
                #self.data.DELTA[(l, str(tuple(k)))] * self.model.h_flow[str(k), p, t] for k in self.data.K_PATHS)
                self.model.h_flow[str(k), p, t] for k in self.data.K_PATHS_L[l] )
        self.model.PathArcRel = Constraint(self.data.EPT, rule=PathArcRule)

        # CAPACITY constraints (compared to 2018) - TEMPORARY
        # the model quickly becomes infeasible when putting such constraints on the model. Should be tailor-made!

        # def CapacityConstraints(model, i,j,m,p,t):
        #    a = (i,j,m)
        #    return self.model.x_flow[a,p,t] <= self.data.buildchain2018[(i,j,m,p)]*2
        # self.model.CapacityConstr = Constraint(self.data.APT,rule=CapacityConstraints)

        # Emissions
        def emissions_rule(model, t):
            return (self.model.total_emissions[t] == sum(
                self.data.E_EMISSIONS[a, p, t] * self.model.x_flow[a, p, t] for p in self.data.P_PRODUCTS
                for a in self.data.A_ARCS))
        self.model.Emissions = Constraint(self.data.TS, rule=emissions_rule) #removed self.data.T_TIME_PERIODS


        "CONSTRAINT NEW/ADDED"

        # Emission limit
        
        def EmissionCapRule(model, t):
            return self.model.total_emissions[t] <= self.data.CO2_CAP[t]/self.factor + self.model.z_emission[t]

        self.model.EmissionCap = Constraint(self.data.TS, rule=EmissionCapRule)
        

        
        #Capacity constraint
        def CapacitatedFlowRule(model,i,j,m,r,t):
            l = (i,j,m,r)
            """return sum(self.model.x_flow[a,p,t] for p in self.data.P_PRODUCTS for f in self.data.FM_FUEL[l[2]]
                        for a in self.data.A_PAIRS[l,f]) <= self.data.Y_BASE_CAP[l]
            + sum(self.model.y_exp_link[l,tau] for tau in self.data.T_TIME_PERIODS if tau <= t)
            #+ self.data.Y_ADD_CAP[l]*sum(self.model.v_edge[(l,tau)] for tau in self.data.T_TIME_PERIODS if tau <= t)"""

            return (sum(self.model.x_flow[a, p, t] for p in self.data.P_PRODUCTS for f in self.data.FM_FUEL[m]
                        # so m = l[2], why not replace that
                        for a in self.data.A_PAIRS[l, f]) <= self.data.Y_BASE_CAP[l] +
                   + self.data.Y_ADD_CAP[l] * sum(self.model.v_edge[l, tau] for tau in self.data.T_TIME_PERIODS if tau < t))
        #A_PAIRS IS REMOVED

        self.model.CapacitatedFlow = Constraint(self.data.ET_RAIL, rule = CapacitatedFlowRule)
        
        
        #Expansion in capacity limit
        def ExpansionLimitRule(model,i,j,m,r):
            l = (i,j,m,r)
            return (sum(self.model.v_edge[(l,t)] for t in self.data.T_TIME_PERIODS) <= self.data.INV_LINK[l])
        self.model.ExpansionCap = Constraint(self.data.E_EDGES_RAIL, rule = ExpansionLimitRule)
        
        
        #Investment in new infrastructure/upgrade
        def InvestmentInfraRule(model,i,j,m,r,f,t):
            l = (i,j,m,r)
            return (sum(self.model.x_flow[a,p,t] for p in self.data.P_PRODUCTS for a in self.data.A_PAIRS[l, f])
                    <= self.data.BIG_M_UPG[l]*sum(self.model.z_inv_upg[l,u,tau]
                    for u in self.data.UF_UPG[f] for tau in self.data.T_TIME_PERIODS if tau < t))
        self.model.InvestmentInfra = Constraint(self.data.LFT_UPG, rule = InvestmentInfraRule)
        
        
        """#Terminal capacity constraint OLD
        def TerminalCapRule(model,i,m,t):
            return (sum(self.model.x_flow[a,p,t] for p in self.data.P_PRODUCTS for a in self.data.A_IN[i,m])
            + sum(self.model.x_flow[a,p,t] for p in self.data.P_PRODUCTS for a in self.data.A_OUT[i,m]) <= self.data.Y_NODE_CAP[i,m]
            + sum(self.model.y_exp_node[i,m,tau] for tau in self.data.T_TIME_PERIODS if tau <= t))
        self.model.TerminalCap = Constraint(self.data.NMT_CAP, rule = TerminalCapRule)"""
        
        #Terminal capacity constraint NEW
        def TerminalCapRule(model, i, m, b, t):
            return(sum(self.model.h_flow[k, p, t] for k in self.data.ORIGIN_PATHS[(i,m)] for p in self.data.PT[b]) + 
                   sum(self.model.h_flow[k, p, t] for k in self.data.DESTINATION_PATHS[(i,m)] for p in self.data.PT[b]) +
                   sum(self.model.h_flow[k,p,t] for k in self.data.TRANSFER_PATHS[(i,m)] for p in self.data.PT[b]) <= 
                   self.data.Y_NODE_CAP[i,m,b]+self.data.Y_ADD_CAP_NODE[i,m,b]*sum(self.model.w_node[i,m,b,tau] for tau in self.data.T_TIME_PERIODS if tau < t))
        self.model.TerminalCap = Constraint(self.data.NMBT_CAP, rule = TerminalCapRule)
        
        #Max terminal capacity expansion NEW -- how many times you can perform a step-wise increase of the capacity
        def TerminalCapExpRule(model, i, m, b):
            return(sum(self.model.w_node[i,m,b,t] for t in self.data.T_TIME_PERIODS) <= self.data.INV_NODE[i,m,b])
        self.model.TerminalCapExp = Constraint(self.data.NMB_CAP, rule = TerminalCapExpRule)

        
        
        
        def ChargingCapArcRule(model, i, j, m, f, r, t):
            l = (i, j, m, r)
            # Must hold for each arc pair a (ijmfr + jimfr) in each time_period t
            # Linear expansion of charging capacity
            return (sum(self.model.x_flow[a, p, t] for p in self.data.P_PRODUCTS
                       for a in self.data.A_PAIRS[l, f]) <= self.data.BASE_CHARGE_CAP[(i,j,m,f,r)] +
                   sum(self.model.y_charge[(i,j,m,f,r,tau)] for tau in self.data.T_TIME_PERIODS if tau <= t))
        self.model.ChargingCapArc = Constraint(self.data.CHARGING_AT, rule=ChargingCapArcRule)

        # NEW CHARGING NODE RESTRICTION
        #
        
        """def ChargingCapNodeRule(model, i, m, f, t):
           # Linear expansion of charging capacity
           return (sum(self.model.x_flow[a, p, t] for p in self.data.P_PRODUCTS
                       for a in self.data.IMF_ARCS[(i, m, f)]) <= self.data.BASE_CHARGE_CAP_NODE[(i,m,f)] +
                   sum(self.model.charge_node[(i,m,f,tau)] for tau in self.data.T_TIME_PERIODS if tau <= t))
        self.model.ChargingCapNode = Constraint(self.data.CHARGING_IMFT, rule=ChargingCapNodeRule)"""


        #def Diesel2020(model, t):
        #    return (sum(self.model.x_flow[(a,p,t)] for p in self.data.P_PRODUCTS
        #        for a in self.data.DIESEL_ROAD) >= sum(self.model.x_flow[(a,p,t)]
        #       for p in self.data.P_PRODUCTS for a in self.data.ARCS_ROAD))
        #self.model.Diesel2020Rate = Constraint(self.data.T_TIME_2020, rule=Diesel2020)
 
        """
        def NonAnticipativityRule(model,a,p):
            return(self.model.x_flow[(a, p, 2020, "average")] == self.model.x_flow[(a, p, 2020, "low")]
            == self.model.x_flow[(a, p, 2020, "high")] == self.model.x_flow[(a, p, 2020, "hydrogen")])
            self.model.NonAnticipativity = Constraint(self.data.AP, rule=NonAnticipativityRule)
            """
        
        #self.scenario_dependent_constraints(self.data.Y_TECH)
        #Technology maturity limit
        self.model.TechMaturityLimit = Constraint(self.data.MFTS_CAP, rule = self.TechMaturityLimitRule)

        return self.model
    
        
    def TechMaturityLimitRule(self,model, m, f, t):
        return (sum(self.model.x_flow[(a, p, t)] for p in self.data.P_PRODUCTS
                    for a in self.data.A_TECH[m, f]) <= self.data.Y_TECH[(m, f, t)])
    
    def update_tech_constraint(self):
        self.model.del_component(self.model.TechMaturityLimit);
        self.model.del_component(self.model.TechMaturityLimit_index);
        self.model.add_component("TechMaturityLimit",Constraint(self.data.MFTS_CAP, rule=self.TechMaturityLimitRule))    

    def solve_model(self):

        self.results = self.opt.solve(self.model, tee=True, symbolic_solver_labels=True,
                                      keepfiles=True)  # , tee=True, symbolic_solver_labels=True, keepfiles=True)

        if (self.results.solver.status == pyomo.opt.SolverStatus.ok) and (
                self.results.solver.termination_condition == pyomo.opt.TerminationCondition.optimal):
            print('the solution is feasible and optimal')
        elif self.results.solver.termination_condition == pyomo.opt.TerminationCondition.infeasible:
            print('the model is infeasible')
            #log_infeasible_constraints(self.model,log_expression=True, log_variables=True)
            #logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)
            #print(value(model.z))

        else:
            print('Solver Status: '), self.results.solver.status
            print('Termination Condition: '), self.results.solver.termination_condition

        print('Solution time: ' + str(self.results.solver.time))
        


        
