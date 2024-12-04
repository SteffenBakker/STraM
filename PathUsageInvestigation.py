import ast
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import chardet


scenario_tree = "FuelScen"     # Options: 4Scen, 9Scen, AllScen

generated_paths_file = "generated_paths_2_modes.csv"     # Select the current created paths file

show_plot = True # Set to True to show the plot

# Load the pickle file

file_path = r"Data//output//SP_" + scenario_tree + ".pickle"
file_path = "Data/Output/FuelScen_carbontaxbase_SP_results.pickle"

with open(file_path, 'rb') as file:
    output_data = pickle.load(file)
    
generated_paths_file = r"Data//SPATIAL//" + generated_paths_file


with open(generated_paths_file, 'rb') as file:
    raw_data = file.read(10000)  # Read a sample
    detected = chardet.detect(raw_data)
    print(detected)

generated_paths = pd.read_csv(generated_paths_file, encoding='ISO-8859-1')



# Access the extracted path flows with path weights
h_path = output_data.h_path

#########
# h_path == 0 are not included #
#########


####################
### Total arcs #####
####################

#Number of generated paths
print("Number of generated paths: ",generated_paths.shape[0])

#Print the number of unique paths with non-zero weight
print("Number of unique paths in the solution: ",h_path['path'].nunique())

#Print the percentage of generated paths that are used in the solution, two decimal places
print("Percentage of generated paths that are used in the solution: ",round(h_path['path'].nunique()/generated_paths.shape[0]*100,2),"%")
#Drop the columns 'variable', 'product', 'scenario' and 'time_period' from the dataframe h_path
h_path = h_path.drop(['variable', 'product', 'scenario', 'time_period'], axis=1)

#Group h_path by path and sum the values
h_path_sum = h_path.groupby(['path']).sum().reset_index() 


#find the number of row with weight close to 0
print("Number of row where the value of the weight transported are <= 0.0001: ",h_path_sum[h_path_sum['weight'] <=0.0001].shape[0])

#Set the paths column in generated_paths to a list of touples
generated_paths['paths'] = generated_paths['paths'].apply(ast.literal_eval)

#join h_path_sum and generated_paths on column 'path' in h_path_sum and the index of generated_paths
h_path_sum = h_path_sum.join(generated_paths, on='path')

#########################
### One or more arcs ####
#########################

#split the h_path_sum dataframe into two dataframes: h_path_sum_1 and h_path_sum_2. Where h_path_sum_1 contains the rows with length of path equal to 1 and h_path_sum_2 contains the rows with length of path equal to 2
h_path_sum_1 = h_path_sum[h_path_sum['paths'].apply(len) == 1]
h_path_sum_2 = h_path_sum[h_path_sum['paths'].apply(len) > 1]

#Print the number of paths with length equal to 1
print("Number of paths with length equal to 1: ",h_path_sum_1.shape[0])

#Print the number of paths with length equal to 2
print("Number of paths with length equal to 2: ",h_path_sum_2.shape[0])

#########################
### One or two modes ####
#########################

#for each row in the column 'paths' in the dataframe h_path_sum_2 check if the first and the last element in the list of touples are the same
h_path_sum_2['uni_mode'] = h_path_sum_2['paths'].apply(lambda x: x[-1][2] == x[0][2])

#split the h_path_sum_2 dataframe into two dataframes: h_path_sum_2_1 and h_path_sum_2_2. Where uni_mode contains the rows with uni_mode equal to True and df contains the rows with uni_mode equal to False
mono_mode = h_path_sum_2[h_path_sum_2['uni_mode'] == True]
di_mode = h_path_sum_2[h_path_sum_2['uni_mode'] == False]

#Print the number of paths with one mode
print("Number of paths with one mode: ",mono_mode.shape[0])

#Print the number of paths with two modes
print("Number of paths with two modes: ",di_mode.shape[0])


###########################
### Investigat 2 modes ####
###########################


city_coordinates = {
    'Oslo': (10.7522, 59.9139),
    'Skien': (9.6087, 59.2049),
    'Kristiansand': (7.9959, 58.1462),
    'Stavanger': (5.7331, 58.9699),
    'Bergen': (5.3221, 60.3913),
    'Førde': (5.8563, 61.4571),
    'Ålesund': (6.1548, 62.4722),
    'Hamar': (11.067, 60.7945),
    'Trondheim': (10.3951, 63.4305),
    'Bodø': (14.4049, 67.2804),
    'Narvik': (17.4277, 68.438),
    'Tromsø': (18.9560, 69.6496),
    'Alta': (23.2707, 69.9663),
    'Umeå': (20.263, 63.8258),
    'Stockholm': (18.0686, 59.3293),
    'Hamburg': (9.9937, 53.5511),
    'World': (-0.118092, 51.509865),  # London used as reference for World
    'JohanSverdrupPlatform': (2.4776, 58.9411)  # Coordinates are illustrative
}

def plot_mode(df):
    G = nx.DiGraph()

    for city, coordinates in city_coordinates.items():
        G.add_node(city, pos=coordinates)

    # Initialize dictionaries to store edge colors and weights for each mode
    mode_colors = {'Sea': 'blue', 'Road': 'green', 'Rail': 'red'}
    mode_weights = {'Sea': [], 'Road': [], 'Rail': []}
    mode_styles = {'Sea': '-', 'Road': '-', 'Rail': '--'}  # Adjust line styles as needed

    # Add edges to the graph based on the paths in the DataFrame
    for paths_list in df['paths']:
        for path in paths_list:
            source, target, mode, direction = path
            G.add_edge(source, target, mode=mode, color=mode_colors[mode], weight=direction)
            mode_weights[mode].append(direction)

    # Get node positions from the 'pos' attribute
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph with node positions based on geographical coordinates
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')
    
    for edge in G.edges(data=True):
        source, target, data = edge
        mode = data['mode']
        if mode == 'Sea':
            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)],
                                   edge_color=mode_colors[mode],
                                   width=direction,
                                   connectionstyle="arc3,rad=0.15")  # Adjust the 'rad' value for curvature
            
        elif mode == 'Rail':
            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)],
                                   edge_color=mode_colors[mode],
                                   width=direction,
                                   connectionstyle="arc3,rad=0.05")  # Adjust the 'rad' value for curvature

        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)],
                                   edge_color=mode_colors[mode],
                                   width=direction,
                                   style=mode_styles[mode])

    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', font_weight='bold')
    
    # Add a legend
    legend_handles = [
        Line2D([0], [0], linestyle='-', color=color, markerfacecolor=color, markersize=10) for color in mode_colors.values()
    ]
    plt.legend(handles=legend_handles, labels=mode_colors.keys(), title='Mode of Travel', loc='upper left')

    # Show the plot
    plt.show()

df = di_mode # Select the dataframe to plot, select from mono_mode or di_mode

if show_plot:
    plot_mode(df)