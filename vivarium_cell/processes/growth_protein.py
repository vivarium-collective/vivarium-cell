from __future__ import absolute_import, division, print_function

import os

import numpy as np

from vivarium.library.units import units
from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium_cell.processes.derive_globals import AVOGADRO


NAME = 'growth_protein'


class GrowthProtein(Process):

    name = NAME
    defaults = {
        'initial_mass': 1339 * units.fg,
        # 'initial_protein': 3.9e7,  # counts of protein
        # the median E. coli protein is 209 amino acids long, and AAs ~ 100 Da
        'protein_mw': 2.09e4 * units.g / units.mol,
        'growth_rate': 0.000275,  # for doubling time about every 2520 seconds
        'global_deriver_key': 'global_deriver',
        'mass_deriver_key': 'mass_deriver',
    }

    def __init__(self, initial_parameters=None):
        if initial_parameters is None:
            initial_parameters = {}

        self.growth_rate = self.or_default(initial_parameters, 'growth_rate')
        self.global_deriver_key = self.or_default(
            initial_parameters, 'global_deriver_key')
        self.mass_deriver_key = self.or_default(
            initial_parameters, 'mass_deriver_key')

        # default state
        # 1000 proteins per fg
        initial_mass = self.or_default(
            initial_parameters, 'initial_mass')
        self.protein_mw = self.or_default(
            initial_parameters, 'protein_mw')
        self.initial_protein = (initial_mass.to('g') / self.protein_mw * AVOGADRO)
        self.divide_protein = self.initial_protein * 2

        parameters = {
            'growth_rate': self.growth_rate}
        parameters.update(initial_parameters)

        super(GrowthProtein, self).__init__(parameters)

    def ports_schema(self):
        return {
            'internal': {
                'protein': {
                    '_default': self.initial_protein,
                    '_divider': 'split',
                    '_emit': True,
                    '_properties': {
                        'mw': self.protein_mw}}},
            'global': {
                'volume': {
                    '_updater': 'set',
                    '_divider': 'split'},
                'divide': {
                    '_default': False,
                    '_updater': 'set'}}}

    def derivers(self):
        return {
            self.mass_deriver_key: {
                'deriver': 'mass_deriver',
                'port_mapping': {
                    'global': 'global'},
                'config': {}},
            self.global_deriver_key: {
                'deriver': 'globals_deriver',
                'port_mapping': {
                    'global': 'global'},
                'config': {}},
        }

    def next_update(self, timestep, states):
        protein = states['internal']['protein']
        total_protein = protein * np.exp(self.parameters['growth_rate'] * timestep)
        new_protein = int(total_protein - protein)
        extra = total_protein - int(total_protein)

        # simulate remainder
        where = np.random.random()
        if where < extra:
            new_protein += 1

        divide = False
        if protein >= self.divide_protein:
            divide = True

        return {
            'internal': {
                'protein': new_protein},
            'global': {
                'divide': divide}}

if __name__ == '__main__':
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    process = GrowthProtein()
    settings = {'total_time': 2520}
    timeseries = simulate_process_in_experiment(process, settings)

    volume_ts = timeseries['global']['volume']
    mass_ts = timeseries['global']['mass']
    print('volume growth: {}'.format(volume_ts[-1] / volume_ts[0]))
    print('mass growth: {}'.format(mass_ts[-1] / mass_ts[0]))

    plot_simulation_output(timeseries, {}, out_dir)
