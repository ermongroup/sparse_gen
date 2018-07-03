"""Some utils for plotting metrics"""
# pylint: disable = C0111

import os
import glob
import numpy as np

import matplotlib
matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode']=True
# matplotlib.rc('font', family='Times New Roman')

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

import sys
# stdout = sys.stdout
# reload(sys)
# sys.setdefaultencoding('utf-8')
# sys.stdout = stdout

def int_or_float(val):
    try:
        return int(val)
    except ValueError:
        return float(val)


def get_figsize(is_save):
    if is_save:
        figsize = [6, 4]
    else:
        figsize = None
    return figsize

def load_if_pickled(pkl_filepath):
    """Load if the pickle file exists. Else return empty dict"""
    if os.path.isfile(pkl_filepath):
        with open(pkl_filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
    else:
        data = {}
    return data

def get_data(expt_dir):
    data = {}
    measurement_losses = load_if_pickled(expt_dir + '/measurement_losses.pkl')
    l2_losses = load_if_pickled(expt_dir + '/l2_losses.pkl')
    l1_losses = load_if_pickled(expt_dir + '/l1_losses.pkl')
    linf_losses = load_if_pickled(expt_dir + '/linf_losses.pkl')
    data = {'measurement': measurement_losses.values(),
            'l2': l2_losses.values(), 'l1': l1_losses.values(), 'linf': linf_losses.values()}

    return data


def get_metrics(expt_dir, name_temp):
    data = get_data(expt_dir)

    metrics = {}
    m_loss_mean = np.mean(data['measurement'])
    m_loss_std = np.std(data['measurement']) / np.sqrt(len(data['measurement']))

    metrics['measurement'] = {'mean': m_loss_mean, 'std': m_loss_std}
    l2_list = [np.sqrt(x) for x in data['l2']]
    data['l2'] = l2_list
    l2_loss_mean = np.mean(data['l2'])
    l2_loss_std = np.std(data['l2']) / np.sqrt(len(data['l2']))
    metrics['l2'] = {'mean':l2_loss_mean, 'std':l2_loss_std}
    
    l1_loss_mean = np.mean(np.array(data['l1']))
    l1_loss_std = np.std(np.array(data['l1'])) / np.sqrt(len(data['l1']))
    metrics['l1'] = {'mean':l1_loss_mean, 'std':l1_loss_std}

    linf_loss_mean = np.mean(data['linf'])
    linf_loss_std = np.std(data['linf']) / np.sqrt(len(data['linf']))
    metrics['linf'] = {'mean':linf_loss_mean, 'std':linf_loss_std}

    return metrics


def get_expt_metrics(expt_dirs, name_temp):
    expt_metrics = {}
    for expt_dir in expt_dirs:
        metrics = get_metrics(expt_dir, name_temp)
        expt_metrics[expt_dir] = metrics
    return expt_metrics


def get_nested_value(dic, field):
    answer = dic
    for key in field:
        answer = answer[key]
    return answer


def find_best(pattern, criterion, retrieve_list, name_temp):
    dirs = glob.glob(pattern)
    metrics = get_expt_metrics(dirs, name_temp)
    best_merit = 1e10
    answer = [None]*len(retrieve_list)
    cur_st = ""
    for dst, val in metrics.iteritems():
        merit = get_nested_value(val, criterion)
        if merit < best_merit:
            cur_st = dst
            best_merit = merit
            for i, field in enumerate(retrieve_list):
                answer[i] = get_nested_value(val, field)

    return answer


def plot(base, regex, criterion, retrieve_list, name_temp):
    keys = map(int_or_float, [a.split('/')[-1] for a in glob.glob(base + '*')])
    means, std_devs = {}, {}
    for i, key in enumerate(keys):
        pattern = base + str(key) + regex
        answer = find_best(pattern, criterion, retrieve_list, name_temp)
        if answer[0] is not None:
            means[key], std_devs[key] = answer
    plot_keys = sorted(means.keys())
    means = np.asarray([means[key] for key in plot_keys])
    std_devs = np.asarray([std_devs[key] for key in plot_keys])
    (_, caps, _) = plt.errorbar(plot_keys, means, yerr=1.96*std_devs,
                                marker='o', markersize=5, capsize=5)
    for cap in caps:
        cap.set_markeredgewidth(1)
