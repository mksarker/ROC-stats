# -*- coding: utf-8 -*-


# **************************************************************
# ROC curve, confidence interval and p-value (bootstrapping)
# 
# v 0.1
#
# Inspired by:
#		Carpenter J, Bithell J: Bootstrap confidence intervals: when, which, what? A practical guide for medical statisticians. Stat Med 2000; 19:1141–64
#		Robin X, Turck N, Hainard A, et al.: pROC: An open-source package for R and S+ to analyze and compare ROC curves. BMC Bioinformatics 2011; 12:77
# 		https://stackoverflow.com/questions/52373318/how-to-compare-roc-auc-scores-of-different-binary-classifiers-and-assess-statist
# 		https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
# 
# © M. Sauthier - 2019 - AGPL v3.0
# **************************************************************


import time, math
import numpy as np
import multiprocessing as mp
from sklearn.metrics import roc_auc_score


def auc_pvalue(y_true, y_prob, y_true2, y_prob2, **kwargs):

	def bootstrap(data, q):
		y_true			= data[0]
		y_prob			= data[1]
		y_true2			= data[2]
		y_prob2			= data[3]
		n_repet			= data[4]
		np.random.seed(data[5])
		paired			= data[6]

		ii, kk			= 0, 0
		aucs_delta		= []
		
		# Split cases and controls stratified
		cases1, controls1 = y_prob[(y_true == 1)], y_prob[(y_true == 0)]
		cases2, controls2 = y_prob2[(y_true2 == 1)], y_prob2[(y_true2 == 0)]

		len_cases1, len_controls1 = len(cases1), len(controls1)
		len_cases2, len_controls2 = len(cases2), len(controls2)

		while ii < n_repet:
			# Generate extraction mask
			mask_cases			= np.random.randint(low=0, high=len_cases1, size=len_cases1)
			mask_controls		= np.random.randint(low=0, high=len_controls1, size=len_controls1)

			# Extract part1
			subcases1_prob		= cases1[mask_cases]
			subcontrols1_prob	= controls1[mask_controls]
		
			subcases1_true		= np.ones(len_cases1)
			subcontrols1_true	= np.zeros(len_controls1)

			auc1 = roc_auc_score(np.concatenate((subcases1_true, subcontrols1_true)), np.concatenate((subcases1_prob, subcontrols1_prob)))


			# If paired is False, re-generate another mask
			if paired is False:
				mask_cases		= np.random.randint(low=0, high=len_cases2, size=len_cases2)
				mask_controls	= np.random.randint(low=0, high=len_controls2, size=len_controls2)


			subcases2_prob		= cases2[mask_cases]
			subcontrols2_prob	= controls2[mask_controls]
		
			subcases2_true		= np.ones(len_cases2)
			subcontrols2_true	= np.zeros(len_controls2)

			auc2 = roc_auc_score(np.concatenate((subcases2_true, subcontrols2_true)), np.concatenate((subcases2_prob, subcontrols2_prob)))

			aucs_delta.append(auc1-auc2)

			ii += 1
		
		q.put(aucs_delta)
		return aucs_delta



	y_true			= np.array(y_true)
	y_prob			= np.array(y_prob)
	y_true2			= np.array(y_true2)
	y_prob2			= np.array(y_prob2)

	n_repet			= kwargs.get('n_repet', 2000)
	n_jobs			= kwargs.get('n_jobs', -1)
	alternative		= kwargs.get('alternative', 'two_sided')
	paired			= kwargs.get('paired', True)
	seed			= kwargs.get('seed', None)

	# Validation at least one positive and one negative in each group
	if np.sum(y_true) in [0, len(y_true)] or np.sum(y_true2) in [0, len(y_true2)]:
		return False

	#Determined if paired data or no
	if paired is None or paired is not False:
		if np.array_equal(y_true,y_true2):
			paired = True
		else:
			paired = False


	# Plannings jobs
	if n_jobs == -1:
		n_jobs = mp.cpu_count()

	# Divide in equal ints
	modulo		= n_repet % n_jobs
	n_block		= int((n_repet - modulo) / n_jobs)
	blocks	= []

	for ii in range(n_jobs):
		add = 0
		if modulo > 0:
			add		= 1
			modulo -= 1

		if seed is not None:
			# Increment seed, if not all processess will be identical
			seed += ii

		blocks.append([y_true, y_prob, y_true2, y_prob2, n_block + add, seed, paired])

	q = mp.Queue()

	procs = []


	for ii in range(n_jobs):
		p = mp.Process(target=bootstrap,	args=(blocks[ii], q))
		procs.append(p)
		# Little pause to allow to empty queue
		time.sleep(0.01)
		p.start()


	# Empty the queue as soon as possible (prevent deadlock)
	# Wait for all worker processes to finish
	results, boot_diffs = [], []

	while len(results) < n_jobs:
		time.sleep(0.001)
		if q.empty() is False:
			results.append(q.get(block=False))

	# deadlock frequent, kill the remaining processes
	for p in procs:
		p.kill()
		# p.join()

	for r in results:
		for r2 in r:
			boot_diffs.append(r2)

	obs_diff	= roc_auc_score(y_true, y_prob) - roc_auc_score(y_true2, y_prob2)

	# As decribe in the pROC module
	z = (obs_diff - 0) / np.std(boot_diffs)

	#Calculating p-value
	if alternative == 'two_sided':
		p_val = 2 * (1 + math.erf(-abs(z) / math.sqrt(2))) / 2

	elif alternative == 'greater':
		p_val = (1 + math.erf(-z / math.sqrt(2))) / 2

	elif alternative == 'less':
		p_val = (1 + math.erf(z / math.sqrt(2))) / 2

	else:
		p_val = None

	return p_val


def auc_ci(y_true, y_prob, **kwargs):

	def bootstrap(data, q):
		y_true			= data[0]
		y_prob			= data[1]
		n_repet			= data[2]
		np.random.seed(data[3])

		ii, jj = 0, 0
		aucs = []
		

		# Split cases and controls STRATIFIED
		cases, controls = y_prob[(y_true == 1)], y_prob[(y_true == 0)]

		len_cases		= len(cases)
		len_controls	= len(controls)

		while ii < n_repet:
			subcases_prob		= np.random.choice(cases, len_cases, replace=True)
			subcontrols_prob	= np.random.choice(controls, len_controls, replace=True)

			subcases_true		= np.ones(len_cases)
			subcontrols_true	= np.zeros(len_controls)

			aucs.append(roc_auc_score(np.concatenate((subcases_true, subcontrols_true)), np.concatenate((subcases_prob, subcontrols_prob))))

			ii += 1

		q.put(aucs)
		return aucs

	y_true 			= np.array(y_true)
	y_prob 			= np.array(y_prob)

	n_repet			= kwargs.get('n_repet', 2000)
	conf_int		= kwargs.get('conf_int', 0.95)
	n_jobs			= kwargs.get('n_jobs', -1)
	seed			= kwargs.get('seed', None)

	# Validation at least one positive and one negative
	if np.sum(y_true) in [0, len(y_true)]:
		return False

	# Plannings jobs
	if n_jobs == -1:
		n_jobs = mp.cpu_count()

	# Divide in equal ints
	modulo		= n_repet % n_jobs
	n_block		= int((n_repet - modulo) / n_jobs)
	blocks		= []

	for ii in range(n_jobs):
		add = 0
		if modulo > 0:
			add		= 1
			modulo -= 1

		if seed is not None:
			# Increment seed, if not all processess will be identical
			seed += ii

		blocks.append([y_true, y_prob, n_block + add, seed])

	q = mp.Queue()

	procs = []

	# print('main process id', os.getpid())

	for ii in range(n_jobs):
		p = mp.Process(target=bootstrap, args=(blocks[ii], q))
		procs.append(p)
		p.start()


	# Empty the queue as soon as possible (prevent deadlock)
	# Wait for all worker processes to finish
	# results = [[],[],[]]
	results, aucs = [], []

	while len(results) < n_jobs:
		time.sleep(0.001)
		if q.empty() is False:
			results.append(q.get(block=False))

	# deadlock frequent, kill the remaining processes
	for p in procs:
		p.kill()
		# p.join()

	# print(results)
	for ad in results:
		for ad2 in ad:
			aucs.append(ad2)

	bounds_auc	= np.quantile(aucs, [0+(1-conf_int)/2, 1-(1-conf_int)/2])
	auc			= roc_auc_score(y_true, y_prob)

	ret = {
		'lowerBound'	: bounds_auc[0],
		'upperBound'	: bounds_auc[1],
		'AUC'			: auc,
		'AUC_mean'		: np.mean(aucs),
		'AUC_sd'		: np.std(aucs),
		'verbose'		: 'AUC {:0.3f} (CI{} %: {:0.3f}-{:0.3f}, N={}) [{} replicates]'.format(
			auc, int(conf_int*100), bounds_auc[0], bounds_auc[1], len(y_true), n_repet)
		}
	return ret