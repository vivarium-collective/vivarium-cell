"""
======================
Inclusion body process
======================
"""

import os
import random

from scipy import constants

from vivarium.library.units import units
from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output


NAME = 'inclusion_body'
AVOGADRO = constants.N_A


def polar_partition(value, front_back):
    """ asymmetric partitioning of inclusion body """
    # front aggregate goes to the front daughter
    if 'front' in front_back:
        aggregate = front_back['front']
        return [aggregate, 0.0]
    # back aggregate goes to the back daughter
    elif 'back' in front_back:
        aggregate = front_back['back']
        return [0.0, aggregate]


class InclusionBody(Process):
    ''' Inclusion Body Process

    Simulates the conversion of biomass into cell damage in inclusion bodies.
    The damage builds up in the front and back poles, with a bistable switch
    that makes the damage aggregate in one pole or the other. Upon division,
    a polar_partition _divider pulls the inclusion body on the front pole into
    one daughter and the inclusion body on the back pole to another.
    '''

    name = NAME
    defaults = {
        'aggregation_rate': 1e-1,
        'damage_rate': 1e-6,
        'unit_mw': AVOGADRO * units.fg / units.mol,
    }

    def __init__(self, initial_parameters=None):
        super(InclusionBody, self).__init__(initial_parameters)
        self.aggregation_rate = self.parameters['aggregation_rate']
        self.damage_rate = self.parameters['damage_rate']

    def initial_state(self, config=None):
        if config is None:
            config = {}
        initial_mass = config.get('initial_mass', 0.0)
        front_back = [0.0, initial_mass]
        random.shuffle(front_back)
        state = {
            'front': {'aggregate': front_back[0]},
            'back': {'aggregate': front_back[1]}}
        if 'molecules' in config:
            state['molecules'] = config['molecules']
        return state

    def ports_schema(self):
        return {
            'front': {
                'aggregate': {
                    '_default': 0.0,
                    '_emit': True,
                    '_divider': {
                        'divider': polar_partition,
                        'topology': {'front': ('aggregate',)}},
                    '_properties': {
                        'mw': self.parameters['unit_mw']
                    },
                }
            },
            'back': {
                'aggregate': {
                    '_default': 0.0,
                    '_emit': True,
                    '_divider': {
                        'divider': polar_partition,
                        'topology': {'back': ('aggregate',)}},
                    '_properties': {
                        'mw': self.parameters['unit_mw']
                    },
                }
            },
            'inclusion_body': {
                '_default': 0.0,
                '_emit': True,
                '_updater': 'set',
                '_divider': 'zero',
            },
            'molecules': {
                '*': {
                    '_default': 0.0,
                }
            },
        }

    def next_update(self, timestep, states):

        # get the states
        front_aggregate = states['front']['aggregate']
        back_aggregate = states['back']['aggregate']
        molecules = states['molecules']
        molecule_mass = sum(molecules.values())

        # existing damage aggregate at front or back, pulled towards pole with larger body
        total_aggregate = front_aggregate + back_aggregate
        if total_aggregate > 0:
            combined = front_aggregate * back_aggregate / (total_aggregate ** 2)
            front_aggregation = self.aggregation_rate * \
                                combined * (front_aggregate - back_aggregate) / total_aggregate
            back_aggregation = self.aggregation_rate * \
                               combined * (back_aggregate - front_aggregate) / total_aggregate
        else:
            front_aggregation = total_aggregate
            back_aggregation = total_aggregate

        # proportionate damage to all molecules
        if molecule_mass > 0:
            total_damage = self.damage_rate * molecule_mass * timestep
            delta_molecules = {
                mol_id: - total_damage * mass / molecule_mass
                for mol_id, mass in molecules.items()}
        else:
            total_damage = molecule_mass
            delta_molecules = {}

        # get total change to front and back aggregates
        polar_damage = total_damage / 2
        delta_front = (front_aggregation + polar_damage) * timestep
        delta_back = (back_aggregation + polar_damage) * timestep

        return {
            'front': {
                'aggregate': delta_front},
            'back': {
                'aggregate': delta_back},
            'inclusion_body': (total_aggregate + total_damage),
            'molecules': delta_molecules,
        }


# functions to configure and run the process
def run_inclusion_body(out_dir='out'):

    # initialize the process by passing initial_parameters
    initial_parameters = {'growth_rate': 1e-1}
    inclusion_body_process = InclusionBody(initial_parameters)

    # get initial state
    initial_state = inclusion_body_process.initial_state({
        'initial_mass': 1.0,
        'molecules': {
            'biomass': 1.0}})

    # run the simulation
    sim_settings = {
        'initial_state': initial_state,
        'total_time': 100}
    output = simulate_process_in_experiment(inclusion_body_process, sim_settings)

    # plot the simulation output
    plot_settings = {}
    plot_simulation_output(output, plot_settings, out_dir)


# run module is run as the main program with python vivarium/process/template_process.py
if __name__ == '__main__':
    # make an output directory to save plots
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_inclusion_body(out_dir)
