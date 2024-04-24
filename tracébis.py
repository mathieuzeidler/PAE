import numpy as np
import matplotlib.pyplot as plt

# Données fournies
p_values = [0.5, 0.75, 1.0]
s = 2

TH_values = [[123705, 123705, 123705], [124845,  116700, 116700], [ 118275, 116685, 113355]]


lower_bounds = []
upper_bounds = []

# Calcul de l'intervalle de confiance à 90% pour chaque point TH
sample_size = len(TH_values)
standard_deviation = np.std(TH_values, ddof=1)
confidence_level = 0.90
z_score = 1.645  # valeur z pour un intervalle de confiance de 90%
standard_error = standard_deviation / np.sqrt(sample_size)

# Calcul de l'intervalle de confiance à 90% pour chaque groupe de points TH
for TH_group in TH_values:
    sample_size = len(TH_group)
    standard_deviation = np.std(TH_group, ddof=1)
    standard_error = standard_deviation / np.sqrt(sample_size)
    margin_of_error = z_score * standard_error
    mean_TH = np.mean(TH_group)
    lower_bound = mean_TH - margin_of_error
    upper_bound = mean_TH + margin_of_error
    lower_bounds.append(lower_bound)
    upper_bounds.append(upper_bound)

# Calcul de l'erreur pour chaque point
lower_errors = [mean_TH - lower_bound for mean_TH, lower_bound in zip([np.mean(TH_group) for TH_group in TH_values], lower_bounds)]
upper_errors = [upper_bound - mean_TH for mean_TH, upper_bound in zip([np.mean(TH_group) for TH_group in TH_values], upper_bounds)]

# Tracer la courbe avec les barres d'erreur
plt.plot(p_values, [np.mean(TH_group) for TH_group in TH_values], marker='o', linestyle='-', color='b', label='Données')
plt.errorbar(p_values, [np.mean(TH_group) for TH_group in TH_values], yerr=[lower_errors, upper_errors], fmt='none', color='black', capsize=5, label='Intervalle de confiance (90%)')

# Ajouter des flèches aux axes
plt.annotate('', xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->'))
plt.annotate('', xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle='->'))

# Augmenter l'espace à gauche du graphique
plt.subplots_adjust(left=0.15)

plt.xlabel('Probabilité (p)')
plt.ylabel('TH')
plt.title('Courbe de TH en fonction de p avec IC (90%) et 2 sources')
plt.grid(True)
plt.legend()
plt.show()