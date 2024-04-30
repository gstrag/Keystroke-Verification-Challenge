import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, det_curve
import scipy as sp
import matplotlib.pyplot as plt

plt.show(block=False)


def compute_eer(labels, scores):
    """
    labels: 1D np.array, 0 = impostor comparison, 1 = genuine comparison
    scores: 1D np.array
    """
    fmr, tmr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnmr = 1 - tmr
    eer_index = np.nanargmin(np.abs((fnmr - fmr)))
    eer_threshold = thresholds[eer_index]
    eer = np.mean((fmr[eer_index], fnmr[eer_index]))
    return eer, eer_threshold


def compute_auc(labels, scores):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :return: np float in range 0-1
    """
    fmr, tmr, _ = roc_curve(labels, scores, pos_label=1)
    return auc(fmr, tmr)


def compute_acc(labels, scores, threshold):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param threshold: threshold value for which to compute accuracy
    :return: np float in range 0-1
    """
    preds = np.where(scores > threshold, 1, 0)
    return accuracy_score(labels, preds)


def compute_fnmr(labels, scores, fmr_t):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param fmr_t: FMR value for which to compute FNMR
    :return: (fnmr for given fmr_t, corresponding threshold value) both as np float in range 0-1
    """
    fmr, tmr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnmr = 1 - tmr
    fnmr_index = np.nanargmin(np.abs((fmr_t - fmr)))
    fnmr_threshold = thresholds[fnmr_index]
    return fnmr[fnmr_index], fnmr_threshold


def compute_rank(labels, scores, N):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param N: rank-N to compute
    :return: np float in range 0-1
    """
    genuine_scores = scores[labels.astype(bool)]
    impostor_scores = scores[np.logical_not(labels.astype(bool))]
    ranks = []
    for scenario in genuine_scores:
        rank_scores = np.insert(impostor_scores, 0, scenario)
        rank_labels = np.insert(np.zeros(len(impostor_scores)), 0, 1)
        rank = np.stack((rank_scores, rank_labels)).T
        rank = rank[np.argsort(rank[:, 0])[::-1]]
        ranks.append(np.sum(rank[:, 1][:N]))
    return np.mean(ranks)


def compute_std(labels, scores, attrs, threshold):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param attrs: 1D np array, same size as labels, containing set of demographic labels as ints
    :param threshold: threshold value for which to compute accuracy
    :return: (np float in range 0-1, ordered accuracy results by demographic group)
    """
    results_by_attr = []
    for attr in sorted(set(attrs)):
        attr_scores = scores[np.where(attrs == attr)]
        attr_labels = labels[np.where(attrs == attr)]
        results_by_attr.append(compute_acc(attr_labels, attr_scores, threshold))
    results_by_attr = np.array(results_by_attr)
    return np.std(results_by_attr, ddof=1), results_by_attr


def compute_ser(labels, scores, attrs, threshold):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param attrs: 1D np array, same size as labels, containing set of demographic labels as ints
    :param threshold: threshold value for which to compute accuracy
    :return: (np float in range 0-1, ordered accuracy results by demographic group)
    """
    results_by_attr = []
    for attr in sorted(set(attrs)):
        attr_scores = scores[np.where(attrs == attr)]
        attr_labels = labels[np.where(attrs == attr)]
        results_by_attr.append(compute_acc(attr_labels, attr_scores, threshold))
    results_by_attr = np.array(results_by_attr)
    return np.max(results_by_attr) / np.min(results_by_attr), results_by_attr


def compute_Gini_coeffs(values):
    """
    :param values: ordered fnmr/fmr results by demographic group
    :return: np float
    """
    n = len(values)
    x_avg = sum(values) / len(values)
    num = 0
    for v1 in values:
        for v2 in values:
            num += abs(v1 - v2)
    return (n / (n - 1)) * (num / (2 * n * n * x_avg))


def compute_garbe(labels, scores, attrs, threshold, alpha=0.5):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param attrs: 1D np array, same size as labels, containing set of demographic labels as ints
    :param threshold: threshold value for which to compute fnmr, fmr
    :param alpha: parameter of GARBE
    :return: np float
    """
    fnmr_by_attr = []
    fmr_by_attr = []
    for attr in set(attrs):
        attr_scores = scores[np.where(attrs == attr)]
        attr_labels = labels[np.where(attrs == attr)]
        genuine_attr_scores = attr_scores[attr_labels.astype(bool)]
        impostor_attr_scores = attr_scores[np.logical_not(attr_labels.astype(bool))]
        fnmr = len(genuine_attr_scores[genuine_attr_scores <= threshold]) / len(genuine_attr_scores)
        fmr = len(impostor_attr_scores[impostor_attr_scores > threshold]) / len(impostor_attr_scores)
        fnmr_by_attr.append(fnmr)
        fmr_by_attr.append(fmr)
    a = compute_Gini_coeffs(np.array(fnmr_by_attr))
    b = compute_Gini_coeffs(np.array(fmr_by_attr))
    garbe = alpha * a + (1 - alpha) * b
    return garbe


def compute_fdr(labels, scores, attrs, threshold, alpha=0.5):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param attrs: 1D np array, same size as labels, containing set of demographic labels as ints
    :param threshold: threshold value for which to compute fnmr, fmr
    :param alpha: parameter of FDR
    :return: np float
    """
    fnmr_by_attr = []
    fmr_by_attr = []
    for attr in set(attrs):
        attr_scores = scores[np.where(attrs == attr)]
        attr_labels = labels[np.where(attrs == attr)]
        genuine_attr_scores = attr_scores[attr_labels.astype(bool)]
        impostor_attr_scores = attr_scores[np.logical_not(attr_labels.astype(bool))]
        fnmr = len(genuine_attr_scores[genuine_attr_scores <= threshold]) / len(genuine_attr_scores)
        fmr = len(impostor_attr_scores[impostor_attr_scores > threshold]) / len(impostor_attr_scores)
        fnmr_by_attr.append(fnmr)
        fmr_by_attr.append(fmr)
    fnmr_by_attr = np.array(fnmr_by_attr)
    fmr_by_attr = np.array(fmr_by_attr)
    a = np.abs(np.max(fnmr_by_attr) - np.min(fnmr_by_attr))
    b = np.abs(np.max(fmr_by_attr) - np.min(fmr_by_attr))
    fdr = 1 - (alpha * a + (1 - alpha) * b)
    return fdr


def compute_ir(labels, scores, attrs, threshold, alpha=0.5):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param attrs: 1D np array, same size as labels, containing set of demographic labels as ints
    :param threshold: threshold value for which to compute fnmr, fmr
    :param alpha: parameter of FDR
    :return: np float
    """
    fnmr_by_attr = []
    fmr_by_attr = []
    for attr in set(attrs):
        attr_scores = scores[np.where(attrs == attr)]
        attr_labels = labels[np.where(attrs == attr)]
        genuine_attr_scores = attr_scores[attr_labels.astype(bool)]
        impostor_attr_scores = attr_scores[np.logical_not(attr_labels.astype(bool))]
        fnmr = len(genuine_attr_scores[genuine_attr_scores <= threshold]) / len(genuine_attr_scores)
        fmr = len(impostor_attr_scores[impostor_attr_scores > threshold]) / len(impostor_attr_scores)
        fnmr_by_attr.append(fnmr)
        fmr_by_attr.append(fmr)
    fnmr_by_attr = np.array(fnmr_by_attr)
    fmr_by_attr = np.array(fmr_by_attr)
    a = np.max(fnmr_by_attr) / np.min(fnmr_by_attr)
    b = np.max(fmr_by_attr) / np.min(fmr_by_attr)
    ir = (a ** alpha) * (b ** (1 - alpha))
    return ir


def compute_sir(scores, comp_type):
    """
    :param scores: 1D np array
    :param comp_type: 2D array with comparison labels for each score value
    :return: (np float, 2D numpy array with SIR matrix)
    """
    categs = np.unique(comp_type, axis=0)
    joint = np.concatenate((scores.reshape(-1, 1), comp_type), axis=1)
    scores_by_demo_group = {}
    for i, categ in enumerate(categs):
        categ_scores = joint[np.logical_and((joint[:, 1:] == categ)[:, 0], (joint[:, 1:] == categ)[:, 1])]
        scores_by_demo_group[tuple(categ)] = categ_scores[:, 0]
    scores_by_demo_group_ = []
    for k in list(scores_by_demo_group.keys()):
        scores_by_demo_group_.append(
            np.concatenate((scores_by_demo_group[(k[0], k[1])], scores_by_demo_group[(k[1], k[0])])))
    mean_scores_by_demo_group = np.reshape(np.array([np.mean(x) for x in scores_by_demo_group_]),
                                           (len(np.unique(categs[:, 0])), len(np.unique(categs[:, 0]))))
    b = np.tril(mean_scores_by_demo_group)
    np.fill_diagonal(b, 0)
    diff_imp_points = b.flatten()
    diff_imp_points = diff_imp_points[diff_imp_points != 0]
    same_imp_points = np.diag(mean_scores_by_demo_group)
    SIR = 100 * (np.mean(same_imp_points) / np.mean(diff_imp_points) - 1)
    return SIR, mean_scores_by_demo_group


def plot_det(ax, labels, scores, name_saved='det_curves', displayed_name='My System', location=''):
    """
    :param ax: matplotlib Axes object
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param name_saved: name of files
    :param displayed_name: name to display in image
    :param location: saving directory
    :return: None
    """
    labels_fs, title_fs, ticks_fs = 14, 14, 12
    ticks = [0.001, 0.01, 0.1, 0.5, 0.90, 0.99, 0.999]
    lowlim = -3.2
    uplim = 1.3
    fmr, fnmr, thresholds = det_curve(labels, scores)
    eer, _ = compute_eer(labels, scores)
    eer = np.round(100 * eer, 2)
    ax.set_xlabel("False Match Rate (%)", fontsize=labels_fs)
    ax.set_ylabel("False Non-Match Rate (%)", fontsize=labels_fs)
    ax.set_title('Detection Error Tradeoff (DET) Curve', fontsize=title_fs)
    ax.plot([-3.5, 3], [-3.5, 3], 'k--')
    tick_locations = sp.stats.norm.ppf(ticks)
    ax.plot([tick_locations[0], tick_locations[0]], [-3.2, 3.2], 'r--')
    ax.plot([tick_locations[1], tick_locations[1]], [-3.2, 3.2], 'r--')
    ax.plot([tick_locations[2], tick_locations[2]], [-3.2, 3.2], 'r--')
    tick_labels = ['{:.0%}'.format(s) if (100 * s).is_integer() else '{:.1%}'.format(s) for s in ticks]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, fontsize=ticks_fs)
    ax.set_xlim(lowlim, uplim)
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels, fontsize=ticks_fs)
    ax.set_ylim(lowlim, uplim)
    ax.plot(sp.stats.norm.ppf(fmr), sp.stats.norm.ppf(fnmr), label=displayed_name + ' - EER ={}%'.format(eer))
    ax.legend(loc="upper right")
    ax.grid('major')
    ax.figure.savefig(location + '{}.png'.format(name_saved), bbox_inches='tight')
    ax.figure.savefig(location + '{}.pdf'.format(name_saved), bbox_inches='tight')


def plot_roc(ax, labels, scores, name_saved='roc_curves', displayed_name='My System', location=''):
    """
    :param ax: matplotlib Axes object
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param name_saved: name of files
    :param displayed_name: name to display in image
    :param location: saving directory
    :return: None
    """
    labels_fs, title_fs, ticks_fs = 14, 14, 12
    fmr, tmr, _ = roc_curve(labels, scores)
    auc_ = np.round(100 * compute_auc(labels, scores), 2)
    ax.plot(fmr, tmr, label=displayed_name + ' - AUC = {}%'.format(auc_))
    ax.plot([0, 1.0], [0, 1.0], 'k--')  # random predictions curve
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    tix = np.arange(0., 1.1, 0.2)
    ax.set_yticks(tix)
    ax.set_yticklabels([str(np.round(scenario * 100)) for scenario in tix], fontsize=ticks_fs)
    ax.set_xticks(tix)
    ax.set_xticklabels([str(np.round(scenario * 100)) for scenario in tix], fontsize=ticks_fs)
    ax.fill_between(fmr, tmr, alpha=.3)
    ax.set_xlabel('False Match Rate (%)', fontsize=labels_fs)
    ax.set_ylabel('True Match Rate (%)', fontsize=labels_fs)
    ax.set_title('Receiving Operator Characteristic (ROC) Curve', fontsize=title_fs)
    ax.legend(loc="lower right")
    ax.grid('major')
    ax.figure.savefig(location + '{}.png'.format(name_saved), bbox_inches='tight')
    ax.figure.savefig(location + '{}.pdf'.format(name_saved), bbox_inches='tight')


def plot_hist(ax, labels, scores, name_saved='score_distributions', displayed_name='My System', location=''):
    """
    :param ax: matplotlib Axes object
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :param name_saved: name of files
    :param displayed_name: name to display in image
    :param location: saving directory
    :return: None
    """
    labels_fs, title_fs, ticks_fs = 14, 14, 12
    genuine_scores = scores[labels.astype(bool)]
    impostor_scores = scores[np.logical_not(labels.astype(bool))]
    _, threshold = compute_eer(labels, scores)
    ax.hist(genuine_scores, bins=int(len(genuine_scores) / 100), histtype='step', label='Genuine Distribution',
            color='green')
    ax.hist(impostor_scores, bins=int(len(impostor_scores) / 100), histtype='step', label='Impostor Distribution',
            color='red')
    ax.axvline(threshold, 0, 100, linestyle='dashed', color='k', label='EER Threshold')
    ax.grid('major')
    ax.set_ylabel('Occurrences', fontsize=labels_fs)
    ax.set_xlabel('Scores', fontsize=labels_fs)
    ax.legend(loc='upper left')
    ax.set_title('Score Distributions ({})'.format(displayed_name), fontsize=title_fs)
    ax.figure.savefig(location + '{}.png'.format(name_saved), bbox_inches='tight')
    ax.figure.savefig(location + '{}.pdf'.format(name_saved), bbox_inches='tight')


def plot_sir_heat_maps(ax, imp_scores, label_names, sir_value, name_saved='sir_heat_map', displayed_name='My System',
                       location=''):
    """
    :param ax: matplotlib Axes object
    :param label_names: list of strings with category names
    :param sir_value: np float
    :param name_saved: name of files
    :param displayed_name: name to display in image
    :param location: saving directory
    :return: None
    """
    N = np.shape(imp_scores)[0]
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(label_names, rotation='vertical')
    ax.set_yticks(np.arange(N))
    ax.set_yticklabels(label_names)
    im = ax.imshow(imp_scores, cmap='YlOrRd')
    ax.set_title('Impostor Score Heat Map (10E-2) of {}. SIR = %1.2f%%'.format(displayed_name) % sir_value)
    for (j, i), label in np.ndenumerate(imp_scores):
        ax.text(i, j, np.round(label * 100, 2), ha='center', va='center')
    ax.figure.savefig(location + '{}.png'.format(name_saved), bbox_inches='tight')
    ax.figure.savefig(location + '{}.pdf'.format(name_saved), bbox_inches='tight')
