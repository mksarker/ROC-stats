# ROC-stats
Compute the confidence interval for the area under the ROC curve (bootstrap stratified method)


# Dependencies

- Numpy
- Scikit-learn


# Usage
```python
# Install ROC_CI in the same repository
# See test.py

from ROC_CI import *

y_prob = np.array([0.21, 0.82, 0.3, 0.1, 0.92, 0.79, 0.82, 0.99, 0.04,0.21, 0.82, 0.3, 0.1, 0.92, 0.79, 0.82, 0.99, 0.04,0.21, 0.82, 0.3, 0.1, 0.92, 0.79, 0.82, 0.99, 0.04,0.21, 0.82, 0.3, 0.1, 0.92, 0.79, 0.82, 0.99, 0.04])
y_prob2	= np.array([0.21, 0.82, 0.1, 0.1, 0.92, 0.79, 0.82, 0.2, 0.04,0.21, 0.9, 0.1, 0.1, 0.92, 0.8, 0.9, 0.1, 0.04,0.21, 0.82, 0.3, 0.1, 0.1, 0.79, 0.82, 0.1, 0.04,0.21, 0.82, 0.3, 0.1, 0.92, 0.79, 0.82, 0.7, 0.04])
y_true = np.array([0,    1,    0,    0,    1,    1,    0,    1,    0   ,0,    1,    0,    0,    1,    1,    0,    1,    0   ,0,    1,    0,    0,    1,    1,    0,    1,    0   ,0,    1,    0,    0,    1,    1,    0,    1,    0   ])


print('AUC1: ',auc_ci(y_true, y_prob, conf_int=0.95, n_repet=2000)['verbose'])
print('AUC2: ',auc_ci(y_true, y_prob2, conf_int=0.95, n_repet=2000)['verbose'])

print()

print('p-value two sided:', auc_pvalue(y_true, y_prob, y_true, y_prob2, n_repet=2000, alternative='two_sided'))
print('p-value greater:', auc_pvalue(y_true, y_prob, y_true, y_prob2, n_repet=2000, alternative='greater'))
print('p-value less:', auc_pvalue(y_true, y_prob, y_true, y_prob2, n_repet=2000, alternative='less'))

print('p-value two sided (rev):', auc_pvalue(y_true, y_prob2, y_true, y_prob, n_repet=2000, alternative='two_sided'))
print('p-value greater (rev):', auc_pvalue(y_true, y_prob2, y_true, y_prob, n_repet=2000, alternative='greater'))
print('p-value less (rev):', auc_pvalue(y_true, y_prob2, y_true, y_prob, n_repet=2000, alternative='less'))

```


# Inspired by

- Carpenter J, Bithell J: Bootstrap confidence intervals: when, which, what? A practical guide for medical statisticians. Stat Med 2000; 19:1141–64
- Robin X, Turck N, Hainard A, et al.: pROC: An open-source package for R and S+ to analyze and compare ROC curves. BMC Bioinformatics 2011; 12:77
- https://stackoverflow.com/questions/52373318/how-to-compare-roc-auc-scores-of-different-binary-classifiers-and-assess-statist
- https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
