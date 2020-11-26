import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats
import math

#retourne un float de 5 decimales
def float0(f):
		return '{:0.5f}'.format(f)

#charge les echantillons venant du fichier .dat
def load_samples():
	samples = defaultdict(lambda: [[] for alg in range(3)])
	with open('QI7_donnee.dat') as file:
		for l in file:
			values = list(map(float, l.split()))
			init_size = int(values[0])
			compr_sizes = values[1:]
			for i in range(3):
				samples[init_size][i].append(compr_sizes[i])
	for size, algs in samples.items():
		for alg in range(3):
			algs[alg].sort()
	return samples

#affiche le graphique de l aspect general des donnees
def show_data_overview(samples):
	f, axes = plt.subplots(3, sharex=True)
	colors = ['r', 'g', 'b']
	for alg in range(3):
		xs = []
		ys = []
		for size, algs in samples.items():
			for sample in algs[alg]:
				xs.append(size)
				ys.append(sample)
		axes[alg].plot(xs, ys, colors[alg] + '.', label='n°' + str(alg + 1))
		axes[alg].legend(loc='upper left')
	plt.suptitle('Allure générale des données')
	plt.xlabel('Taille initiale')
	f.text(0.04, 0.5, 'Taille après compression', va='center', rotation='vertical')
	plt.show()

#affiche les histogrammes des donnees taille par taille
def show_hists(samples):
	colors = ['r', 'g', 'b']
	for size, algs in samples.items():
		f, axes = plt.subplots(3, 1, sharex=True)
		for alg in range(3):
			data = algs[alg]
			nb_bars = 10
			axes[alg].hist(data, nb_bars, color=colors[alg], label='n°' + str(alg + 1))
			axes[alg].legend(loc='upper left')
		plt.suptitle('Taille initiale : ' + str(size))
		plt.xlabel('Taille après compression')
		f.text(0.04, 0.5, 'Nombre de données par intervalle', va='center', rotation='vertical')
		plt.show()

#retourne la moyenne
def mean(data):
	return sum(data) / len(data)

#retourne la variance biaisee des donnees
def variance(data):
	total = 0
	mean_data = mean(data)
	for x in data:
		total += (x - mean_data)**2
	return total / (len(data) - 1)

#affiche les resultats du test de Kolmogorov-Smirnov
def show_kolmogorov_smirnov_test(samples):
	print('====== Kolmogorov-Smirnov test ======')
	for size, algs in samples.items():
		print('- Size : ' + str(size))
		for alg in range(3):
			sup = 0
			F0 = 0
			data = algs[alg]
			count = len(data)
			mean_data = mean(data)
			SD_data = math.sqrt(variance(data))
			for value in data:
				norm_value = (value - mean_data) / SD_data
				F0 += 1 / count
				Fn = stats.norm.cdf(norm_value)
				diff = abs(Fn - F0)
				sup = max(sup, diff)
			D_alpha = 0.134 # pour n = 100 et alpha = 0.05
			D_n = sup
			is_gaussian = D_n < D_alpha
			print('    Algo ' + str(alg + 1) + ' : ' + str(is_gaussian))
			print('moyenne : '+str(mean_data))
			print('variance : '+str(SD_data))
			print('D_n : '+str(D_n))
	print('=====================')

#retourne le resultat du test de moyenne
def confidence_interval_test(samples):
	result = defaultdict(dict)
	for size, algs in samples.items():
		to_comp = [(i, j) for i in range(3) for j in range(i + 1, 3)]
		for (alg1, alg2) in to_comp:
			data1 = algs[alg1]
			data2 = algs[alg2]
			n1 = len(data1)
			n2 = len(data2)
			num = abs(mean(data1) - mean(data2))			
			den = math.sqrt((variance(data1)**2 / n1) + \
						            (variance(data2)**2 / n2))
			test = num / den
			Z = 1.96 #distribution normale standard d ordre 1 - 0.05/2
			#a decommenter pour resultat du test de moyenne
			#print(test) 
			compatible = test > Z
			result[size][(alg1, alg2)] = compatible
	return result

#retourne le resultat du test de Fisher
def fisher_test(samples):
	result = defaultdict(dict)
	for size, algs in samples.items():
		to_comp = [(i, j) for i in range(3) for j in range(i + 1, 3)]
		for (alg1, alg2) in to_comp:
			data1 = algs[alg1]
			data2 = algs[alg2]
			variance1 = variance(data1)
			variance2 = variance(data2)
			n1 = len(data1)
			n2 = len(data2)
			num = (n1 / (n1 - 1)) * variance1
			den = (n2 / (n2 - 1)) * variance2
			T = num / den
			if variance1 <= variance2:
				T = 1 / T
			F = 1.48 # Fisher-Snedecor with v1 = 100 and v2 = 100 and order 0.05/2
			#a decommenter pour resultat du test de Fisher
			#print(T)
			compatible = T < F
			result[size][(alg1, alg2)] = compatible
	return result

#affiche la compatibilité entre les algo
def show_compatibility(samples):
	print('============= Compatible algos =============')
	confidence_interval = confidence_interval_test(samples)
	fisher = fisher_test(samples)
	for size in samples:
		print('- Size : ' + str(size))
		to_comp = [(i, j) for i in range(3) for j in range(i + 1, 3)]
		for (alg1, alg2) in to_comp:
			compatible_mean = confidence_interval[size][(alg1, alg2)]
			compatible_variance = fisher[size][(alg1, alg2)]
			compatible = compatible_variance and compatible_mean
			print('    Algos ' + str(alg1 + 1) + ' and ' + str(alg2 + 1) + \
			    ' : (' + str(compatible_mean) + ', ' + \
				 str(compatible_variance) + ') => ' + str(compatible))
	print('============================================')

samples = load_samples()

show_data_overview(samples)
show_hists(samples)
show_kolmogorov_smirnov_test(samples)
show_compatibility(samples)