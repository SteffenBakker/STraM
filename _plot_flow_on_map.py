"""
In this file we plot the resulting flows in the model on a map of Norway
"""

import warnings
warnings.filterwarnings("ignore")

# IMPORTS

import numpy as np
import pandas as pd
import pickle
from mpl_toolkits.basemap import Basemap #for creating the background map
import matplotlib.pyplot as plt #for plotting on top of the background map
import matplotlib.patches as patches #import library for fancy arrows/edges

# DEFINE FUNCTIONS 

# function that processes and aggregates flows
def process_and_aggregate_flows(x_flow, b_flow, sel_scenario, sel_time_period, sel_product,all_products):
    """
    Process model output (x_flow and b_flow), aggregate the flow per edge, and output a dataframe.
    All for a selected scenario and time period

    INPUT
    x_flow:           dataframe with flow of goods
    b_flow:           dataframe with balancing flow (empty vehicles)
    sel_scenario:     scenario name or "average"
    sel_time_period:  time period in [2022, 2026, 2030, 2040, 2050]
    sel_product:      product type in [...] or "all"
    
    OUTPUT
    df_flow:          dataframe with aggregated flows   
    """

    #copy all flows into one dataframe: product flow and balancing flow
    all_flow = x_flow[["from", "to", "mode", "fuel", "product", "scenario", "time_period", "weight"]]
    # add empty vehicle flow
    all_flow = pd.concat([all_flow,
                            b_flow[["from", "to", "mode", "fuel", "scenario", "time_period", "weight"]] ])
    
    prods= all_products
    if sel_product != "all":
        if sel_product == "all_no_dry_bulk":
            prods = prods - ["Dry bulk"]
        else:
            prods = [sel_product]

    #create lists that will store aggregate flows (these will be the columns of df_flow)
    arcs = []
    flows = []
    flows_road = []
    flows_sea = []
    flows_rail = []

    #scenario list and counter in order to take average when necessary
    all_scenarios = []

    #list all nodes clockwise, to make sure sea edges curve in the right direction (HARDCODED)
    nodes_sea_order = ["Umeå", "Stockholm", "Hamar", "Oslo", "Skien", "Kristiansand", "Stavanger", 
                        "Bergen", "Førde","Ålesund", "Trondheim", "Bodø", "Tromsø","Narvik", "Alta",
                        "Hamburg", "World", "JohanSverdrupPlatform"] #HARDCODED

    #add arcs and corresponding flow to the right lists
    #Note: I use the word arc, but we treat them as edges. That is, we look at undirectional flow by aggregating over both directions
    for index, row in all_flow.iterrows():
        if row["time_period"] == sel_time_period:
            if sel_scenario == "average" or row["scenario"] == sel_scenario: #if "average", we add everything and divide by number of scenarios at the end
                if row["product"] in prods: # check if we have a right product category
                    #add scenario to list if not observed yet (for taking average)
                    if row["scenario"] not in all_scenarios:
                        all_scenarios.append(row["scenario"])
                    #temporarily store current arc and its opposite
                    cur_arc = (row["from"], row["to"])
                    cur_cra = (row["to"], row["from"]) #opposite arc
                    #check if new arc
                    if cur_arc not in arcs and cur_cra not in arcs: #new arc
                        #determine direction of arc based on nodes_sea_order and append the arc
                        from_order = nodes_sea_order.index(row["from"]) 
                        to_order = nodes_sea_order.index(row["to"])
                        if from_order < to_order:
                            arcs.append(cur_arc) #append forward arc
                        else:
                            arcs.append(cur_cra) #append backward arc
                        #add zero values for the corresponding flows (initialization)
                        flows.append(0.0)
                        flows_road.append(0.0)
                        flows_sea.append(0.0)
                        flows_rail.append(0.0)
                    #find index of current arc (or cra) in list "arcs"
                    cur_arc_ind = None
                    if cur_arc in arcs:
                        cur_arc_ind = arcs.index(cur_arc)
                    elif cur_cra in arcs:
                        cur_arc_ind = arcs.index(cur_cra)
                    #store corresponding flows in lists
                    flows[cur_arc_ind] += row["weight"]
                    if row["mode"] == "Road":
                        flows_road[cur_arc_ind] += row["weight"]
                    elif row["mode"] == "Sea":
                        flows_sea[cur_arc_ind] += row["weight"]
                    elif row["mode"] == "Rail":
                        flows_rail[cur_arc_ind] += row["weight"]

    #divide everything by number of scenarios if we have selected sel_scenario="average"
    if sel_scenario == "average":
        num_scenarios = len(all_scenarios)
        flows = [(1.0/num_scenarios) * f for f in flows]
        flows_road = [(1.0/num_scenarios) * f for f in flows_road]
        flows_sea = [(1.0/num_scenarios) * f for f in flows_sea]
        flows_rail = [(1.0/num_scenarios) * f for f in flows_rail]

    #store aggregate flows in a dataframe
    df_flow = pd.DataFrame()
    df_flow["arc"] = arcs
    df_flow["orig"] = [""]*len(arcs)
    df_flow["dest"] = [""]*len(arcs)
    df_flow["flow"] = flows
    df_flow["flow_road"] = flows_road
    df_flow["flow_sea"] = flows_sea
    df_flow["flow_rail"] = flows_rail

    #fix origins and destinations
    for i in range(len(df_flow)):
        df_flow.orig[i] = str(df_flow.arc[i][0])
        df_flow.dest[i] = str(df_flow.arc[i][1])

    #return a dataframe with aggregated flows
    return df_flow

# function that computes difference in flows between two years
def compute_flow_differences(x_flow, b_flow, sel_scenario, sel_time_period_before, sel_time_period_after, sel_product,all_products):   
    """
    Compute the difference in flows between two selected years. Outputs a dataframe with these differences
    """
    #create dataframes for before and after year
    df_flow_before = process_and_aggregate_flows(x_flow, b_flow, sel_scenario, sel_time_period_before, sel_product, all_products)
    df_flow_after = process_and_aggregate_flows(x_flow, b_flow, sel_scenario, sel_time_period_after, sel_product, all_products)

    #initialize lists that will be columns of df_flow_diff
    arcs_diff = []
    orig_diff = []
    dest_diff = []
    flow_diff = []
    flow_road_diff = []
    flow_sea_diff = []
    flow_rail_diff = []

    # add flows from after
    for index, row in df_flow_after.iterrows():
        #add arc if it is new
        if row["arc"] not in arcs_diff:
            arcs_diff.append(row["arc"])
            orig_diff.append(row["orig"])
            dest_diff.append(row["dest"])
            flow_diff.append(0.0)
            flow_road_diff.append(0.0)
            flow_sea_diff.append(0.0)
            flow_rail_diff.append(0.0)
        # find arc index
        cur_index = arcs_diff.index(row["arc"])
        # add flows at right index
        flow_diff[cur_index] += row["flow"]
        flow_road_diff[cur_index] += row["flow_road"]
        flow_sea_diff[cur_index] += row["flow_sea"]
        flow_rail_diff[cur_index] += row["flow_rail"]

    # subtract flows from before
    for index, row in df_flow_before.iterrows():
        #add arc if it is new
        if row["arc"] not in arcs_diff:
            arcs_diff.append(row["arc"])
            orig_diff.append(row["orig"])
            dest_diff.append(row["dest"])
            flow_diff.append(0.0)
            flow_road_diff.append(0.0)
            flow_sea_diff.append(0.0)
            flow_rail_diff.append(0.0)
        # find arc index
        cur_index = arcs_diff.index(row["arc"])
        # add flows at right index
        flow_diff[cur_index] -= row["flow"]
        flow_road_diff[cur_index] -= row["flow_road"]
        flow_sea_diff[cur_index] -= row["flow_sea"]
        flow_rail_diff[cur_index] -= row["flow_rail"]

    # store differences in a dataframe
    df_flow_diff = pd.DataFrame()
    df_flow_diff["arc"] = arcs_diff
    df_flow_diff["orig"] = orig_diff
    df_flow_diff["dest"] = dest_diff
    df_flow_diff["flow"] = flow_diff
    df_flow_diff["flow_road"] = flow_road_diff
    df_flow_diff["flow_sea"] = flow_sea_diff
    df_flow_diff["flow_rail"] = flow_rail_diff

    # return dataframe with flow differences between the two years per edge
    return df_flow_diff

# function that plots flow on the map
def plot_flow_on_map(df_flow, base_data, flow_variant, mode_variant, sel_product, plot_overseas=True, plot_up_north=True, show_fig=True, save_fig=False):    
    """
    Create a plot on the map of Norway with all the flows

    INPUT
    df_flow:        dataframe with aggregated flows per edge (split out by mode)
    base_data:      base model data (only used to extract N_NODES)
    flow_variant:   type of flow input, i.e., absolute flow or difference between two years; choose form ["flow", "diff"]
    mode_variant:   what mode to plot; choose from ["road", "sea", "rail", "all", "total"]
    plot_overseas:  indicate whether to plot flow to oversees nodes ("Kontinentalsokkelen", "Europa", and "Verden")
    plot_up_north:  indicate whether to plot flow to nodes up north ("Bodø" and "Tromsø")
    show_fig:       indicate whether to show the figure
    save_fig:       indicate whether to save the figure (filename is determined automatically)
    
    OUTPUT
    figure that is shown and/or saved to disk if requested
    """

    fig = plt.figure(figsize=(6,3))
    ax = plt.axes([0,0,1,1])

    ####################################
    # a. Extract nodes and coordinates

    #extract nodes from base_data
    N_NODES = base_data.N_NODES
    lats = base_data.N_LATITUDE_PLOT
    longs = base_data.N_LONGITUDE_PLOT
    node_xy_offset = base_data.N_COORD_OFFSETS

    node_colors = ["black"]*len(N_NODES)     

    nodes_sea_order = ["Umeå", "Stockholm", "Hamar", "Oslo", "Skien", "Kristiansand", "Stavanger", 
                        "Bergen", "Førde","Ålesund", "Trondheim", "Bodø", "Tromsø","Narvik", "Alta",
                        "Hamburg", "World", "JohanSverdrupPlatform"] #HARDCODED

    ####################
    # b. Build a map

    # create underlying figure/axis (to get rid of whitespace)
    fig = plt.figure(figsize=(6,3))
    ax = plt.axes([0,0,1,1])

    #draw the basic map including country borders
    map = Basemap(llcrnrlon=1, urcrnrlon=29, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='aeqd', lat_0=63.4, lon_0=10.4) # Azimuthal Equidistant Projection
    # map = Basemap(llcrnrlon=1, urcrnrlon=29, llcrnrlat=55, urcrnrlat=70, resolution='i', projection='tmerc', lat_0=0, lon_0=0) # mercator projection
    map.drawmapboundary(fill_color='paleturquoise')
    map.fillcontinents(color='lightgrey', lake_color='paleturquoise')
    map.drawcoastlines(linewidth=0.2)
    map.drawcountries(linewidth=0.2)

    #draw nodes on the map
    node_x, node_y = map(list(longs.values()), list(lats.values()))
    coordinate_mapping={N_NODES[i]:(node_x[i],node_y[i]) for i in range(len(N_NODES))}
    map.scatter(node_x, node_y, color=node_colors, zorder=100)


    ##########################
    # c. Plot flow in the map

    #arrow settings
    tail_width_dict = {"road":10, "rail":10, "sea":20, "all":10, "diff":5} #base tail width for different plotting variants
    min_tail_width = 1 #minimum width of any drawn edge
    #select maximum tail width:
    tail_width_base = 0 #initialize
    if flow_variant == "flow":
        tail_width_base = tail_width_dict[mode_variant]
    elif flow_variant == "diff":
        tail_width_base = tail_width_dict["diff"]
    head_with = 0.01
    head_length = 0.01
    base_curvature = 0.2
    #arrow settings for the different modes
    mode_color_dict = {"road":"violet", "sea":"blue", "rail":"saddlebrown", "total":"black"}
    mode_linestyle_dict = {"road":"-", "sea":"-", "rail":(0, (1, 5)), "total":"-"}
    curvature_fact_dict = {"road":0, "sea":-1.5, "rail":+1.5, "total":0}
    zorder_dict = {"road":30, "sea":20, "rail":40, "total":20}
    # arrow settings for direction of change (for "diff" option)
    dir_color_dict = {"increase":"seagreen", "decrease":"red"}


    # compute maximum and total flows over all edges (for scaling purposes)
    #   note: use absolute value to deal with flow_diff if we use that plotting option
    max_flow = max(abs(df_flow["flow"]))
    max_flow_road = max(abs(df_flow["flow_road"]))
    max_flow_sea = max(abs(df_flow["flow_sea"]))
    max_flow_rail = max(abs(df_flow["flow_rail"]))
    total_flow = sum(abs(df_flow["flow"]))
    total_flow_road = sum(abs(df_flow["flow_road"]))
    total_flow_sea = sum(abs(df_flow["flow_sea"]))
    total_flow_rail = sum(abs(df_flow["flow_rail"]))
    #power_of_ten = 4 #10^power_of_ten is the reference for the max_flow
    #store in dictionaries
    total_flow_dict = {"road":total_flow_road, "sea":total_flow_sea, "rail":total_flow_rail, "total":total_flow, "all":total_flow}
    max_flow_dict = {"road":max_flow_road, "sea":max_flow_sea, "rail":max_flow_rail, "total":max_flow, "all":max_flow}
    

    #iterate over egdes
    for index, row in df_flow.iterrows():
        #store current origin and destinations + indices
        cur_orig = row["orig"]
        cur_dest = row["dest"]
        cur_orig_index = N_NODES.index(cur_orig)
        cur_dest_index = N_NODES.index(cur_dest)
        #check if it is a long distance (temporarily don't plot those to avoid cluttering)
        overseas = False
        up_north = False
        if cur_orig in ["JohanSverdrupPlatform", "Hamburg", "World"] or cur_dest in ["JohanSverdrupPlatform", "Hamburg", "World"]:
            overseas = True
        if cur_orig in ["Bodø", "Tromsø","Narvik","Alta"] or cur_dest in ["Bodø", "Tromsø","Narvik","Alta"]:
            up_north = True
        #check mode variant
        if mode_variant == "all": #we will plot all modes in one figure
            #create dictionary that stores all the flows
            flow_dict = {"road":row["flow_road"], "sea":row["flow_sea"], "rail":row["flow_rail"]}
            #loop over the three modes
            for cur_mode in ["sea", "road", "rail"]:
                #extract information the current mode
                cur_flow = abs(flow_dict[cur_mode]) # take absolute value to deal with "diff" version
                cur_sign = np.sign(flow_dict[cur_mode]) # sign of flow
                cur_total_flow = total_flow_dict[mode_variant]
                cur_max_flow = max_flow_dict[mode_variant]
                curvature_factor = curvature_fact_dict[cur_mode] #indicates in what direction the arc should bend
                # determine plot color
                cur_color = "k" # initialize color at black
                if flow_variant == "flow":
                    cur_color = mode_color_dict[cur_mode]
                elif flow_variant == "diff":    
                    cur_direction = ""
                    if cur_sign >= 0.0:
                        cur_direction = "increase"
                    else:
                        cur_direction = "decrease"
                    cur_color = dir_color_dict[cur_direction]
                #create new arc
                if cur_flow > 0.001*cur_total_flow: #only plot an arc if we have significant flow (at least 0.1% of total flow for the relevant mode)
                #if 10.0 ** power_of_ten * cur_flow/cur_max_flow > 1.0: #only plot an arc if we have significant flow (at least 0.1% of total flow for the relevant mode)
                    new_arc = patches.FancyArrowPatch(
                        (node_x[cur_orig_index], node_y[cur_orig_index]),  #origin coordinates
                        (node_x[cur_dest_index], node_y[cur_dest_index]),  #destination coordinates
                        connectionstyle=f"arc3,rad={base_curvature * curvature_factor}", #curvature of the edge
                        linewidth = max(tail_width_base * cur_flow/cur_max_flow, min_tail_width),
                        
                        linestyle=mode_linestyle_dict[cur_mode],
                        color=cur_color,
                        zorder = zorder_dict[cur_mode]
                        )    
                    if ((not overseas) or (overseas and plot_overseas)) and ((not up_north) or (up_north and plot_up_north)): #only add the arc if we want to plot it
                        #add the arc to the plot
                        plt.gca().add_patch(new_arc) 
        else: #we only plot one mode
            #get current flow and related information
            cur_flow = 0.0
            if mode_variant == "total":
                cur_flow = abs(row["flow"]) # take absolute value to deal with "diff" version
                cur_sign = np.sign(row["flow"]) # sign of flow
            elif mode_variant == "road":
                cur_flow = abs(row["flow_road"])
                cur_sign = np.sign(row["flow_road"])
            elif mode_variant == "sea":
                cur_flow = abs(row["flow_sea"])
                cur_sign = np.sign(row["flow_sea"])
            elif mode_variant == "rail":
                cur_flow = abs(row["flow_rail"])
                cur_sign = np.sign(row["flow_rail"])
            cur_total_flow = total_flow_dict[mode_variant]
            cur_max_flow = max_flow_dict[mode_variant]
            curvature_factor = curvature_fact_dict[mode_variant] #indicates in what direction the arc should bend
            # determine plot color
            cur_color = "k" # initialize color at black
            if flow_variant == "flow":
                cur_color = mode_color_dict[mode_variant]
            elif flow_variant == "diff":    
                cur_direction = ""
                if cur_sign >= 0.0:
                    cur_direction = "increase"
                else:
                    cur_direction = "decrease"
                cur_color = dir_color_dict[cur_direction]
            #create new arc
            if cur_flow > 0.000001*cur_total_flow: #only plot an arc if we have significant flow (at least 0.1% of total flow for the relevant mode)
                new_arc = patches.FancyArrowPatch(
                    (node_x[cur_orig_index], node_y[cur_orig_index]), 
                    (node_x[cur_dest_index], node_y[cur_dest_index]), 
                    connectionstyle=f"arc3,rad={base_curvature * curvature_factor}",
                    #arrowstyle=f"Simple, tail_width={tail_width_base * cur_flow/cur_max_flow}, head_width={head_with}, head_length={head_length}", #tail width: constant times normalized flow
                    linewidth = max(tail_width_base * cur_flow/cur_max_flow, min_tail_width),
                    linestyle=mode_linestyle_dict[mode_variant],
                    color=cur_color,
                    zorder = zorder_dict[mode_variant]
                    )    
                if ((not overseas) or (overseas and plot_overseas)) and ((not up_north) or (up_north and plot_up_north)): #only add the arc if we want to plot it
                    #add the arc to the plot
                    plt.gca().add_patch(new_arc)
        
    ###############################
    # d. Show and save the figure

    #set size
    scale = 1.3
    plot_width = 5 #in inches
    plot_height = scale * plot_width
    plt.gcf().set_size_inches(plot_width, plot_height, forward=True) #TODO: FIND THE RIGH TSIZE
    #save figure
    if save_fig:
        filename = f"Data/Plots/flow_plot_{sel_time_period}_{sel_scenario}_{flow_variant}_{sel_product}.png"
        plt.savefig(filename, bbox_inches="tight")
    #show figure
    if show_fig:
        plt.show()

# function that processes data and makes a flow plot in one go
def process_and_plot_flow(output, base_data, mode_variant, sel_scenario, sel_time_period, sel_product="all", plot_overseas=True, plot_up_north=True, show_fig=True, save_fig=False):
    # compute flow 
    print("Computing flow...")
    df_flow = process_and_aggregate_flows(output.x_flow, output.b_flow, sel_scenario, sel_time_period, sel_product, all_products=base_data.P_PRODUCTS)

    # make plot
    print("Making plot...")
    plot_flow_on_map(df_flow, base_data, "flow", mode_variant, sel_product, plot_overseas, plot_up_north, show_fig, save_fig)

# function that processes data and makes a diff plot in one go
def process_and_plot_diff(output, base_data, mode_variant, sel_scenario, sel_time_period_before, sel_time_period_after, sel_product = "all", plot_overseas=True, plot_up_north=True, show_fig=True, save_fig=False):    
    
    # compute flow differences
    print("Computing flow differences...")
    df_flow_diff = compute_flow_differences(output.x_flow, output.b_flow, sel_scenario, sel_time_period_before, sel_time_period_after, sel_product, all_products=base_data.P_PRODUCTS)

    # make plot
    print("Making plot...")
    plot_flow_on_map(df_flow_diff, base_data, "diff", mode_variant, sel_product, plot_overseas, plot_up_north, show_fig, save_fig)



################################################


# RUN ANALYSIS

# Read model output
analyses_type = 'SP' # EV , EEV, 'SP
scenario_type = "FuelScen" # 4Scen
with open(r'Data/base_data/' + scenario_type + ".pickle", 'rb') as data_file:
    base_data = pickle.load(data_file)
with open(r'Data/Output/'+analyses_type + "_" + scenario_type + ".pickle", 'rb') as output_file:
    output = pickle.load(output_file)



# MULTIPLE PLOTS

#"SETTINGS"

mode_variants = ["all"] # ["road", "sea", "rail", "all", "total"]
sel_scenarios = ["PP","BB","OO"]   # Now it is only a combination of two scenarios. So, choose "BB" or "PB"
sel_time_periods = base_data.T_TIME_PERIODS #2023
sel_products = base_data.P_PRODUCTS + ['all','all_no_dry_bulk']  #"Container (slow)" # "Dry bulk" # any product group or "all"

plot_overseas = True
plot_up_north = True
show_fig = False
save_fig = True

for mode_variant in mode_variants:
    for sel_scenario in sel_scenarios:
        for sel_time_period in sel_time_periods:
            for sel_product in sel_products:
                process_and_plot_flow(output, base_data, mode_variant, sel_scenario, sel_time_period, sel_product, plot_overseas, plot_up_north, show_fig, save_fig)



#SINGLE PLOTS

# 1. Make flow plots

# Choose settings
mode_variant = "all" # ["road", "sea", "rail", "all", "total"]
sel_scenario = "BB"   # Now it is only a combination of two scenarios. So, choose "BB" or "PB"
sel_time_period = 2023 #
sel_product = "Container (slow)" # "Dry bulk" # any product group or "all"  OR " all_no_dry_bulk"
#Dry bulk, Liquid bulk, Container (fast), Container (slow), Break bulk (fast), Break bulk (slow), Neo bulk

plot_overseas = True
plot_up_north = True
show_fig = True
save_fig = True

# Make plot
if False:
    process_and_plot_flow(output, base_data, mode_variant, sel_scenario, sel_time_period, sel_product, plot_overseas, plot_up_north, show_fig, save_fig)


# 2. Make difference plots

# Choose settings
mode_variant = "all" # ["road", "sea", "rail", "all", "total"]
sel_scenario = "BB"  # Now it is only a combination of two scenarios. So, choose "BB" or "PB"
sel_time_period_before = 2023
sel_time_period_after = 2050
sel_product = "all" # any product group or "all"
plot_overseas = True
plot_up_north = True
show_fig = True
save_fig = True

# Make plot
if False:
    process_and_plot_diff(output, base_data, mode_variant, sel_scenario, sel_time_period_before, sel_time_period_after, sel_product, plot_overseas, plot_up_north, show_fig, save_fig)


