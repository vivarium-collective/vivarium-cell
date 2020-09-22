from __future__ import absolute_import, division, print_function

import os

import matplotlib.pyplot as plt

from vivarium.library.units import units
from vivarium_cell.processes.derive_globals import AVOGADRO


def plot_exchanges(timeseries, plot_config, out_dir='out', filename='exchanges'):
    # plot exchanges with the environment

    nAvogadro = AVOGADRO
    external_ts = timeseries['external']
    internal_ts = timeseries['internal']
    global_ts = timeseries['global']

    env_volume = plot_config['environment']['volume']
    legend_on = plot_config.get('legend', True)
    aspect_ratio = plot_config.get('aspect_ratio', 1)

    # pull volume and mass out from internal
    volume = global_ts.pop('volume') * units.fL
    mass = global_ts.pop('mass')

    # conversion factor
    mmol_to_counts = [nAvogadro.to('1/mmol') * vol.to('L') for vol in volume]

    # plot results
    cols = 1
    rows = 3

    width = 5
    height = width / aspect_ratio
    plt.figure(figsize=(width, height))

    # define subplots
    ax1 = plt.subplot(rows, cols, 1)
    ax2 = plt.subplot(rows, cols, 2)
    ax3 = plt.subplot(rows, cols, 3)

    # plot external state
    for mol_id, series in external_ts.items():
        ax1.plot(series, label=mol_id)
    if legend_on:
        ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=2)
    ax1.title.set_text('environment: {}'.format(env_volume))
    ax1.set_ylabel('external \n (mM)')
    ax1.set_yscale('log')

    # plot internal counts
    for mol_id, counts_series in internal_ts.items():
        # conc_series = [(count / conversion).to('mmol/L').magnitude
        #    for count, conversion in zip(counts_series, mmol_to_counts)]
        ax2.plot(counts_series, label=mol_id)

    if legend_on:
        ax2.legend(loc='center left', bbox_to_anchor=(1.6, 0.5), ncol=3)
    # ax2.title.set_text('internal metabolites')
    ax2.set_ylabel('metabolites \n (counts)')
    ax2.set_yscale('log')

    # plot mass
    ax3.plot(mass, label='mass')
    ax3.set_ylabel('mass \n (fg)')

    # adjust axes
    for axis in [ax1, ax2, ax3]:
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xlabel('time (s)')

    # save figure
    fig_path = os.path.join(out_dir, filename)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(fig_path, bbox_inches='tight')

# energy carriers in BiGG models
BiGG_energy_carriers = [
    'atp_c',
    'gtp_c',
    'nad_c',
    'nadp_c',
    'fad_c',
]

def energy_synthesis_plot(timeseries, settings, out_dir, figname='energy_use'):
    # plot the synthesis of energy carriers in BiGG model output
    energy_reactions = settings.get('reactions', {})
    saved_reactions = timeseries['reactions']
    time_vec = timeseries['time']

    # get each energy carrier's total flux
    carrier_use = {}
    for reaction_id, coeffs in energy_reactions.items():
        reaction_ts = saved_reactions[reaction_id]

        for mol_id, coeff in coeffs.items():

            # save if energy carrier is used
            if coeff < 0:
                added_flux = [-coeff*ts for ts in reaction_ts]
                if mol_id not in carrier_use:
                    carrier_use[mol_id] = added_flux
                else:
                    carrier_use[mol_id] = [
                        sum(x) for x in zip(carrier_use[mol_id], added_flux)]

    # make the figure
    n_cols = 1
    n_rows = 1
    fig = plt.figure(figsize=(n_cols * 6, n_rows * 2))
    grid = plt.GridSpec(n_rows, n_cols)

    # first subplot
    ax = fig.add_subplot(grid[0, 0])
    for mol_id, series in carrier_use.items():
        ax.plot(time_vec, series, label=mol_id)
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_title('energy use')
    ax.set_xlabel('time ($s$)')
    ax.set_ylabel('$(mmol*L^{{{}}}*s^{{{}}})$'.format(-1, -1))  # TODO -- use unit schema in figures

    # save figure
    fig_path = os.path.join(out_dir, figname)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig(fig_path, bbox_inches='tight')
