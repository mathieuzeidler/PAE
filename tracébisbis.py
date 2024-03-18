import numpy as np
import matplotlib.pyplot as plt

# Données fournies
p_values = [0.5, 0.75, 1.0]
s = 2 # Nombre de sources

# Données à rentrer 

P_values = [[8239, 8239, 8239], [8321, 7779, 7779], [7882, 7729, 7556]]

D_values = [[0.727268, 0.727268, 0.727268], [0.809328, 1.137317, 1.137317], [1.092405, 0.882805, 0.722627]]

percent_P_values = [[80.710511 ,  80.710511, 80.710511], [81.870942, 76.491987,  76.491987], [77.723016, 76.531064, 74.431203]]

percent_L_values = [[19.289489, 19.289489, 19.289489], [18.129058, 23.508013, 23.508013], [22.276984, 23.468936, 25.568797]]

TH_values = [[123705, 123705, 123705], [124845,  116700, 116700], [ 118275, 116685, 113355]]

J_values = [[1.463623, 1.463623, 1.463623], [0.471624, 2.458906, 2.458906], [3.292968, 1.645932, 1.489564]]


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


