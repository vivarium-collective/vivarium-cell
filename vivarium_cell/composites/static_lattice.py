from __future__ import absolute_import, division, print_function

import os

from vivarium.core.process import Generator
from vivarium.core.composition import (
    compartment_in_experiment,
    COMPARTMENT_OUT_DIR,
)

# processes
from vivarium_cell.processes.multibody_physics import (
    Multibody,
    agent_body_config,
)
from vivarium_cell.plots.multibody_physics import plot_snapshots
from vivarium_cell.processes.static_field import StaticField

NAME = 'static_lattice'


class StaticLattice(Generator):

    defaults = {
        'multibody': {
            'bounds': [10, 10],
            'agents': {}
        },
        'field': {
            'molecules': ['glc'],
            'n_bins': [10, 10],
            'bounds': [10, 10],
        }
    }

    def __init__(self, config):
        super(StaticLattice, self).__init__(config)

    def generate_processes(self, config=None):
        multibody = Multibody(config['multibody'])
        field = StaticField(config['field'])

        return {
            'multibody': multibody,
            'field': field}

    def generate_topology(self, config=None):
        return {
            'multibody': {
                'agents': ('agents',)},
            'field': {
                'agents': ('agents',)}}


def get_static_lattice_config(config={}):
    bounds = config.get('bounds', [25, 25])
    molecules = config.get('molecules', ['glc'])
    n_bins = config.get('n_bins', tuple(bounds))
    center = config.get('center', [0.5, 0.5])
    deviation = config.get('deviation', 5)
    n_agents = config.get('n_agents', 1)
    agent_ids = [str(agent_id) for agent_id in range(n_agents)]

    # multibody config
    mbp_config = {
        # 'animate': True,
        'jitter_force': 1e2,
        'bounds': bounds}
    body_config = {
        'bounds': bounds,
        'agent_ids': agent_ids}
    mbp_config['agents'] = agent_body_config(body_config)

    # field config
    field_config = {
        'molecules': molecules,
        'n_bins': n_bins,
        'bounds': bounds,
        # 'agents': agents,
        'gradient': {
            'type': 'exponential',
            'molecules': {
                mol_id: {
                    'center': [0.0, 0.0],
                    'base': 1 + 1e-1}
                for mol_id in molecules}},
        # 'initial_state': {
        #     mol_id: np.ones((n_bins[0], n_bins[1]))
        #     for mol_id in molecules}
    }

    return {
        'bounds': bounds,
        'multibody': mbp_config,
        'field': field_config}

def test_static_lattice(config=get_static_lattice_config(), end_time=10):

    # configure the compartment
    compartment = StaticLattice(config)

    # configure experiment
    experiment_settings = {
        'compartment': config}
    experiment = compartment_in_experiment(
        compartment,
        experiment_settings)

    # run experiment
    timestep = 1
    time = 0
    while time < end_time:
        experiment.update(timestep)
        time += timestep
    return experiment.emitter.get_data()



if __name__ == '__main__':
    out_dir = os.path.join(COMPARTMENT_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    config = get_static_lattice_config()
    data = test_static_lattice(config, 40)

