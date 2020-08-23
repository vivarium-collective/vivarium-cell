from __future__ import absolute_import, division, print_function

import os

import matplotlib.pyplot as plt

from vivarium.core.composition import set_axes
from vivarium.library.dict_utils import get_value_from_path


def plot_diauxic_shift(timeseries, settings={}, out_dir='out'):
    external_path = settings.get('external_path', ('environment',))
    internal_path = settings.get('internal_path', ('cytoplasm',))
    internal_counts_path = settings.get('internal_counts_path', ('cytoplasm_counts',))
    reactions_path = settings.get('reactions_path', ('reactions',))
    global_path = settings.get('global_path', ('global',))

    time = [t/60 for t in timeseries['time']]  # convert to minutes

    environment = get_value_from_path(timeseries, external_path)
    cell = get_value_from_path(timeseries, internal_path)
    cell_counts = get_value_from_path(timeseries, internal_counts_path)
    reactions = get_value_from_path(timeseries, reactions_path)
    globals = get_value_from_path(timeseries, global_path)

    # environment
    lactose = environment['lcts_e']
    glucose = environment['glc__D_e']

    # internal
    LacY = cell['LacY']
    lacy_RNA = cell['lacy_RNA']
    LacY_counts = cell_counts['LacY']
    lacy_RNA_counts = cell_counts['lacy_RNA']

    # reactions
    glc_exchange = reactions['EX_glc__D_e']
    lac_exchange = reactions['EX_lcts_e']

    # global
    mass = globals['mass']

    # settings
    environment_volume = settings.get('environment_volume')

    n_cols = 2
    n_rows = 4

    # make figure and plot
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 1.5))
    grid = plt.GridSpec(n_rows, n_cols)

    ax1 = fig.add_subplot(grid[0, 0])  # grid is (row, column)
    ax1.plot(time, glucose, label='glucose')
    ax1.plot(time, lactose, label='lactose')
    set_axes(ax1)
    ax1.title.set_text('environment, volume = {} L'.format(environment_volume))
    ax1.set_ylabel('(mM)')
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    ax2 = fig.add_subplot(grid[1, 0])  # grid is (row, column)
    ax2.plot(time, lacy_RNA, label='lacy_RNA')
    ax2.plot(time, LacY, label='LacY')
    set_axes(ax2)
    ax2.title.set_text('internal')
    ax2.set_ylabel('(mM)')
    ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    ax3 = fig.add_subplot(grid[2, 0])  # grid is (row, column)
    ax3.plot(time, mass, label='mass')
    set_axes(ax3, True)
    ax3.title.set_text('global')
    ax3.set_ylabel('(fg)')
    ax3.set_xlabel('time (min)')
    ax3.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    ax4 = fig.add_subplot(grid[0, 1])  # grid is (row, column)
    ax4.plot(time, glc_exchange, label='glucose exchange')
    ax4.plot(time, lac_exchange, label='lactose exchange')
    set_axes(ax4, True)
    ax4.title.set_text('flux'.format(environment_volume))
    ax4.set_xlabel('time (min)')
    ax4.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # save figure
    fig_path = os.path.join(out_dir, 'diauxic_shift')
    plt.subplots_adjust(wspace=0.6, hspace=0.5)
    plt.savefig(fig_path, bbox_inches='tight')
