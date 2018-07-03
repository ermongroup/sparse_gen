import glob
import os
import matplotlib
matplotlib.use('agg')
#uncomment to use latex
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode']=True
# matplotlib.rc('font', family='Times New Roman') 
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import metrics_utils
def save_plot(is_save, save_path):
    if is_save:
        pdf = PdfPages(save_path)
        pdf.savefig(bbox_inches='tight')
        pdf.close()

matplotlib.rcParams.update({'font.size': 15})
is_save = True
figsize = metrics_utils.get_figsize(is_save)

## Define what to plot
criterion = ['l2', 'mean']
retrieve_list_inf = [['linf', 'mean'], ['linf', 'std']]
retrieve_list_1 = [['l1', 'mean'], ['l1', 'std']]
retrieve_list_2 = [['l2', 'mean'], ['l2', 'std']]
legend_base_regexs_mnist_transfer = [
    ('LASSO'  , './estimated/mnist/full-input/gaussian/0.1/', '/lasso/*'),
            ('VAE'    , './estimated/mnist/transfer-full/gaussian/0.1/', '/vae/*'),
        ('Sparse-VAE', './estimated/mnist/transfer-full/gaussian/0.1/', '/vae_l1/*')
]

legend_base_regexs_omniglot_transfer = [

    ('LASSO'  ,'./estimated/omniglot/full-input/gaussian/0.1/', '/lasso/*'),
    ('VAE'    , './estimated/omniglot/transfer-full/gaussian/0.1/', '/vae/*'),
        ('Sparse-VAE', './estimated/omniglot/transfer-full/gaussian/0.1/', '/vae_l1/*')

]


legend_base_regexs_omniglot = [
    ('LASSO'  ,'./estimated/omniglot/full-input/gaussian/0.1/', '/lasso/*'),
  ('VAE'    , './estimated/omniglot/full-input/gaussian/0.1/', '/vae/*'),
        ('Sparse-VAE', './estimated/omniglot/full-input/gaussian/0.1/', '/vae_l1/*')
]

legend_base_regexs_mnist = [
 ('LASSO'  , './estimated/mnist/full-input/gaussian/0.1/', '/lasso/*'),
    ('VAE'    , './estimated/mnist/full-input/gaussian/0.1/', '/vae/*'),
        ('Sparse-VAE', './estimated/mnist/full-input/gaussian/0.1/', '/vae_l1/*')
]

legend_base_regexs_celebA = [
    ('Lasso (DCT)'     , '../estimated/celebA/full-input/gaussian/0.01/', '/lasso-dct/*'),
    ('Lasso (Wavelet)' , '../estimated/celebA/full-input/gaussian/0.01/', '/lasso-wavelet/*'),
    ('DCGAN'       , '../estimated/celebA/full-input/gaussian/0.01/', '/dcgan/*'),
    ('Sparse-DCGAN (Standard)', '../estimated/celebA/full-input/gaussian/0.01/', '/dcgan_l1/*'),
    ('Sparse-DCGAN (Wavelet)', '../estimated/celebA/full-input/gaussian/0.01/', '/dcgan_l1_wavelet/*')
]

graph_type_list = ["mnist_transfer", "mnist", "omniglot", "omniglot_transfer", "celebA"]
legend_regexs_list = [legend_base_regexs_mnist_transfer, legend_base_regexs_mnist, legend_base_regexs_omniglot, legend_base_regexs_omniglot_transfer, legend_base_regexs_celebA]

## Plot_l1
for type_name, legend_base_regexs in zip(graph_type_list, legend_regexs_list):
    plt.figure(figsize=figsize)
    legends = []
    for legend, base, regex in legend_base_regexs:
        metrics_utils.plot(base, regex, criterion, retrieve_list_1, legend)
        legends.append(legend)



    save_path = './'+type_name +'_l1.pdf'

    ## Prettify
    # axis
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xscale("log", nonposx='clip')
    plt.gca().set_xlim([9, 800])

    # labels, ticks, titles
    ticks = [50, 100, 200, 300, 400, 500, 750] # 
    labels = [50, 100, 200, 300, 400, 500, 750] # 
    plt.xticks(ticks, labels, rotation=90)
    plt.ylabel('l1 reconstruction error (per pixel)')
    plt.xlabel('Number of measurements')
    ax=plt.subplot()
    ax.set_xlim(50, 750)
    # Legends
    plt.legend(legends, fontsize=12.5)

    # Saving
    save_plot(is_save, save_path)
    plt.clf()

## Plot_l2
for type_name, legend_base_regexs in zip(graph_type_list, legend_regexs_list):
    plt.figure(figsize=figsize)
    legends = []
    for legend, base, regex in legend_base_regexs:
        metrics_utils.plot(base, regex, criterion, retrieve_list_2, legend)
        legends.append(legend)



    save_path = './'+type_name+ '_l2.pdf'

    ## Prettify
    # axis
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xscale("log", nonposx='clip')
    plt.gca().set_xlim([9, 800])

    # labels, ticks, titles
    ticks = [50, 100, 200, 300, 400, 500, 750] # 
    labels = [50, 100, 200, 300, 400, 500, 750] # 
    plt.xticks(ticks, labels, rotation=90)
    plt.ylabel('l2 reconstruction error (per pixel)')
    plt.xlabel('Number of measurements')
    ax=plt.subplot()
    ax.set_xlim(50, 750)
    # Legends
    plt.legend(legends, fontsize=12.5)

    # Saving
    save_plot(is_save, save_path)
    plt.clf()


## Plot_linf
for type_name, legend_base_regexs in zip(graph_type_list, legend_regexs_list):
    plt.figure(figsize=figsize)
    legends = []
    for legend, base, regex in legend_base_regexs:
        metrics_utils.plot(base, regex, criterion, retrieve_list_inf, legend)
        legends.append(legend)



    save_path = './'+type_name+ '_linf.pdf'

    ## Prettify
    # axis
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xscale("log", nonposx='clip')
    plt.gca().set_xlim([9, 800])

    # labels, ticks, titles
    ticks = [50, 100, 200, 300, 400, 500, 750] # 
    labels = [50, 100, 200, 300, 400, 500, 750] # 
    plt.xticks(ticks, labels, rotation=90)
    plt.ylabel('linf reconstruction error')
    plt.xlabel('Number of measurements')
    ax=plt.subplot()
    ax.set_xlim(50, 750)
    # Legends
    plt.legend(legends, fontsize=12.5)

    # Saving
    save_plot(is_save, save_path)
    plt.clf()