import numpy as np
import matplotlib.pyplot as plt


# Données fournies
p_values = [0.5, 0.75, 1.0]
s = 2
TH_values = [123705, 124845, 118275]

# Calcul de l'intervalle de confiance à 90% pour chaque point TH
sample_size = len(TH_values)
standard_deviation = np.std(TH_values, ddof=1)
confidence_level = 0.90
z_score = 1.645  # valeur z pour un intervalle de confiance de 90%
standard_error = standard_deviation / np.sqrt(sample_size)

lower_bounds = []
upper_bounds = []

for TH in TH_values:
    margin_of_error = z_score * standard_error
    lower_bound = TH - margin_of_error
    upper_bound = TH + margin_of_error
    lower_bounds.append(lower_bound)
    upper_bounds.append(upper_bound)

# Calcul de l'erreur pour chaque point
lower_errors = [TH - lower_bound for TH, lower_bound in zip(TH_values, lower_bounds)]
upper_errors = [upper_bound - TH for TH, upper_bound in zip(TH_values, upper_bounds)]

# Tracer la courbe avec les barres d'erreur
plt.plot(p_values, TH_values, marker='o', linestyle='-', color='b', label='Données')
plt.errorbar(p_values, TH_values, yerr=[lower_errors, upper_errors], fmt='none', color='black', capsize=5, label='Intervalle de confiance (90%)')

# Augmenter l'espace à gauche du graphique
plt.subplots_adjust(left=0.15)

plt.xlabel('Probabilité (p)')
plt.ylabel('TH')
plt.title('Courbe de TH en fonction de p avec IC (90%) et sources')
plt.grid(True)
plt.legend()
plt.show()