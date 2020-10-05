"""
======================
Inclusion body process
======================
"""
import os
import random

from vivarium.library.units import units
from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)
from vivarium.library.dict_utils import deep_merge
from vivarium.plots.simulation_output import plot_simulation_output


NAME = 'inclusion_body'

def polar_partition(value, front_back):
    """ asymmetric partitioning of inclusion body """
    # front body goes to the front daughter
    if 'front' in front_back:
        inclusion_body = front_back['front']
        return [inclusion_body, 0]
    # back body goes to the back daughter
    elif 'back' in front_back:
        inclusion_body = front_back['back']
        return [0, inclusion_body]


class InclusionBody(Process):
    '''
    This mock process provides a basic template that can be used for a new process
    '''

    # declare default parameters as class variables
    defaults = {
        'aggregation': 1e-1,
        'absorption': 1e-3,
        'unit_mw': 2.09e4 * units.g / units.mol,
        'molecules_list': [],
    }

    def __init__(self, initial_parameters=None):
        super(InclusionBody, self).__init__(initial_parameters)
        self.aggregation = self.parameters['aggregation']
        self.absorption = self.parameters['absorption']

    def initial_state(self, config=None):
        if config is None:
            config = {}
        front_back = [0.0, 1.0]
        random.shuffle(front_back)
        state = {
            'global': {
                'mass': 1339 * units.fg},
            'inclusion_body': {
                'front': front_back[0],
                'back': front_back[1]
            }
        }
        return deep_merge(state, config)

    def ports_schema(self):
        return {
            'inclusion_mass': {
                'front': {
                    '_default': 0.0,
                    '_emit': True,
                    '_divider': {
                        'divider': polar_partition,
                        'topology': {'front': ('front',)}},
                    '_properties': {
                        'mw': self.parameters['unit_mw']},
                },
                'back': {
                    '_default': 0.0,
                    '_emit': True,
                    '_divider': {
                        'divider': polar_partition,
                        'topology': {'back': ('back',)}},
                    '_properties': {
                        'mw': self.parameters['unit_mw']},
                },
                'combined': {
                    '_default': 0.0,
                    '_emit': True,
                    '_updater': 'set',
                    '_divider': 'zero',
                }
            },
            'molecules': {
                mol_id: {
                    '_default': 0.0,
                    '_emit': True,
                    # '_divider': 'split',
                    '_properties': {
                        'mw': self.parameters['unit_mw']}
                }
                for mol_id in self.parameters['molecules_list']
            },
            'global': {
                'mass': {
                    '_emit': True,
                    '_default': 0.0 * units.fg,
                    '_updater': 'set',
                    '_divider': 'split'
                },
            }
        }

    def next_update(self, timestep, states):
        # get the states
        front_body = states['inclusion_mass']['front']
        back_body = states['inclusion_mass']['back']
        molecules = states['molecules']
        molecule_mass = sum(molecules.values())

        total_body = front_body + back_body
        if total_body > 0:
            front_ratio = front_body / total_body
            back_ratio = back_body / total_body
            front_aggregation = self.aggregation * back_ratio * front_ratio * (front_ratio - back_ratio) * total_body
            back_aggregation = self.aggregation * back_ratio * front_ratio * (back_ratio - front_ratio) * total_body
        else:
            front_aggregation = total_body
            back_aggregation = total_body

        if molecule_mass > 0:
            total_growth = self.absorption * molecule_mass
            half_growth = total_growth / 2
            delta_molecules = {
                mol_id: - self.absorption * mass / molecule_mass
                for mol_id, mass in molecules.items()}
        else:
            half_growth = molecule_mass
            delta_molecules = {}

        delta_front = (front_aggregation + half_growth) * timestep
        delta_back = (back_aggregation + half_growth) * timestep

        return {
            'inclusion_mass': {
                'front': delta_front,
                'back': delta_back,
                'combined': front_body + back_body + delta_front + delta_back
            },
            'molecules': delta_molecules
        }


# functions to configure and run the process
def run_inclusion_body(out_dir='out'):

    # initialize the process by passing initial_parameters
    initial_parameters = {
        'molecules_list': ['glucose'],
        'growth_rate': 1e-1,
    }
    inclusion_body_process = InclusionBody(initial_parameters)

    # get initial state
    initial_state = inclusion_body_process.initial_state({
        'molecules': {
            'glucose': 1.0}})

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