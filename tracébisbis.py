import numpy as np
import matplotlib.pyplot as plt

# Données fournies
p_values = [0.5, 0.75, 1.0]

############################################################################################

# Pour 2 sources / x,y = 500 / n=50
#s = 2 # Nombre de sources 

#P_values = [[8239, 8239, 8239], [8321, 7779, 7779], [7882, 7729, 7556]]

#D_values = [[0.727268, 0.727268, 0.727268], [0.809328, 1.137317, 1.137317], [1.092405, 0.882805, 0.722627]]

#percent_P_values = [[80.710511 ,  80.710511, 80.710511], [81.870942, 76.491987,  76.491987], [77.723016, 76.531064, 74.431203]]

#percent_L_values = [[19.289489, 19.289489, 19.289489], [18.129058, 23.508013, 23.508013], [22.276984, 23.468936, 25.568797]]

#TH_values = [[123705, 123705, 123705], [124845,  116700, 116700], [ 118275, 116685, 113355]]

#J_values = [[1.463623, 1.463623, 1.463623], [0.471624, 2.458906, 2.458906], [3.292968, 1.645932, 1.489564]]



# Pour 3 sources / x,y = 500 / n=50
#s = 3 # Nombre de sources 

#P_values = [[5648, 5928, 5675], [5802, 1013, 1013], [5020, 5631, 5631]]

#D_values = [[0.729904, 0.551846, 0.712351], [0.508018, 1.021665, 1.021665], [0.691450, 0.901616, 0.901616]]

#percent_P_values = [[62.866195,  70.276757, 77.221895], [64.731607, 109.667872,  109.667872], [53.840216, 72.338454, 72.338454]]

#percent_L_values = [[37.133805, 29.723243, 22.778105], [35.268393, -9.667872, -9.667872], [46.159784, 27.661546, 27.661546]]

#TH_values = [[96727.500000, 107602.500000, 118080.000000], [101025.00000,  25012.500000, 25012.500000], [ 83700.000000, 111247.500000, 111247.500000]]

#J_values = [[0.069180, 0.385035, 5.294564], [5.013951, 4.227733, 4.227733], [0.039972, 2.653847, 2.653847]]


# Pour 4 sources / x,y = 500 / n=50
#s = 4 # Nombre de sources 

#P_values = [[4936, 5285, 5432], [3310, 5553, 4050], [1299, 3350, 3891]]

#D_values = [[0.352361, 0.312996, 0.515994], [0.991320, 0.586117, 0.286620], [1.073641, 0.624094, 0.569092]]

#percent_P_values = [[55.260314,  62.409010, 59.772194], [54.920054, 63.972207, 46.610666], [89.858824, 60.315828, 61.874038]]

#percent_L_values = [[44.739686, 37.590990, 40.227806], [45.079946, 36.027793, 53.389334], [10.141176, 39.684172, 38.125962]]

#TH_values = [[84382.500000, 95167.500000, 91702.50000], [51780.000000, 98055.000000, 71580.0000], [28642.500000, 61590.000000,78397.500000]]

#J_values = [[0.028490, 0.208083, 0.044726], [0.016241, 0.093057, 0.030473], [1.772728, 0.010874, 1.582890]]


############################################################################################


# Pour 2 sources / x,y = 1000 / n=100
#s = 2 # Nombre de sources 

#P_values = [[6909, 6909, 6909], [7155, 6894, 6894], [8342, 8342, 7836]]

#D_values = [[0.795731, 0.795731, 0.795731], [0.746329, 0.755738, 0.755738], [0.895511, 0.895511, 0.758118]]

#percent_P_values = [[68.358664,  68.358664, 68.358664], [70.449983, 67.990574, 67.990574], [81.651063, 81.651063, 76.960522]]

#percent_L_values = [[31.641336, 31.641336, 31.641336], [29.550017, 32.009426, 32.009426], [18.348937, 18.348937, 23.039478]]

#TH_values = [[104235, 104235, 104235], [107910, 103867.5, 103867.5], [125587.5, 125587.5,118575]]

#J_values = [[0.790917, 0.790917, 0.790917], [0.908454, 1.116926, 1.116926], [0.728594, 0.728594, 0.419592]]


# Pour 3 sources / x,y = 1000 / n=100
#s = 3 # Nombre de sources 

#P_values = [[5377, 4708, 5446], [5326, 4750, 4750], [4855, 5935, 5888]]

#D_values = [[0.881540, 0.596907, 0.676345], [0.796663, 0.523062, 0.523062], [0.857343, 0.663614, 0.758323]]

#percent_P_values = [[64.042033,  51.557280, 68.310409], [68.880400, 70.820478, 70.820478], [58.111941, 72.260240, 58.715641]]

#percent_L_values = [[35.957967, 48.442720, 31.689591], [31.119600, 29.179522, 29.179522], [41.888059, 27.739760, 41.284359]]

#TH_values = [[98272.5, 78712.5, 105037.5], [105480, 92962.5, 92962.5], [75300, 110872.5,90517.5]]

#J_values = [[0.764743, 4.856679, 0.031015], [0.222253, 2.517907, 2.517907], [2.970780, 1.562416, 0.021693]]


# Pour 4 sources / x,y = 1000 / n=100
s = 4 # Nombre de sources 

P_values = [[4048, 5471, 3450], [3941, 5113, 5276], [5341, 4617, 5269]]

D_values = [[0.727877, 0.448799, 0.721572], [0.470127, 0.490832, 0.607952], [0.495833, 0.504309, 0.451561]]

percent_P_values = [[67.636520, 56.202631, 71.273840], [68.165573, 53.712280, 65.846289], [59.672945, 58.108042, 62.791152]]

percent_L_values = [[32.363480, 43.797369, 28.726160], [31.834427, 46.287720, 34.153711], [40.327055, 41.891958, 37.208848]]

TH_values = [[70785, 86205, 68107.5], [78675, 82635, 101332.5], [91957.5, 88500, 96442.5]]

J_values = [[2.427083, 0.022832, 0.027801], [1.894120, 0.052358, 0.031204], [0.101893, 0.449809, 1.561598]]



# Liste de toutes les valeurs
all_values = [TH_values, P_values, percent_P_values, percent_L_values, D_values, J_values]

# Liste des noms des valeurs
names = ['TH', 'P', '%P', '%L', 'D', 'J']

# Fonction pour calculer l'intervalle de confiance
def calculate_confidence_interval(values):
    lower_bounds = []
    upper_bounds = []
    for group in values:
        sample_size = len(group)
        standard_deviation = np.std(group, ddof=1)
        confidence_level = 0.90
        z_score = 1.645  # valeur z pour un intervalle de confiance de 90%
        standard_error = standard_deviation / np.sqrt(sample_size)
        mean_value = np.mean(group)
        lower_bound = mean_value - z_score * standard_error
        upper_bound = mean_value + z_score * standard_error
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
    return lower_bounds, upper_bounds

for values, name in zip(all_values, names):
    plt.figure()  # Créer une nouvelle figure
    mean_values = [np.mean(group) for group in values]
    lower_bounds, upper_bounds = calculate_confidence_interval(values)
    yerr = [np.array(mean_values) - np.array(lower_bounds), np.array(upper_bounds) - np.array(mean_values)]
    plt.errorbar(p_values, mean_values, yerr=yerr, fmt='o')
    # Ajouter des barres horizontales pour l'intervalle de confiance
    for x, lower, upper in zip(p_values, lower_bounds, upper_bounds):
        plt.hlines(y=lower, xmin=x-0.01, xmax=x+0.01, color='black')
        plt.hlines(y=upper, xmin=x-0.01, xmax=x+0.01, color='black')
    # Augmenter l'espace à gauche du graphique
    plt.subplots_adjust(left=0.15)
    plt.xlabel('Probabilité (p)')
    plt.ylabel(name)
    plt.title(f'Curve of {name} as a function of p with CI (90%) for {s} sources')
    # plt.grid(True)  # Commenter ou supprimer cette ligne pour enlever le quadrillage
    plt.show()


