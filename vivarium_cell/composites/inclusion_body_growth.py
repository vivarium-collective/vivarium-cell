from __future__ import absolute_import, division, print_function

import os
import copy

from vivarium.library.units import units
from vivarium.core.process import Generator
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    COMPARTMENT_OUT_DIR,
)
from vivarium.plots.agents_multigen import plot_agents_multigen

# processes
from vivarium.processes.meta_division import MetaDivision
from vivarium_cell.processes.inclusion_body import InclusionBody
from vivarium_cell.processes.growth_protein import GrowthProtein
from vivarium_cell.processes.growth import Growth

from vivarium.library.dict_utils import deep_merge


NAME = 'inclusion_body_growth'

class InclusionBodyGrowth(Generator):

    defaults = {
        'growth_rate': 0.006,  # very fast growth
        'boundary_path': ('boundary',),
        'agents_path': ('..', '..', 'agents',),
        'daughter_path': tuple()}

    def __init__(self, config):
        self.config = config
        for key, value in self.defaults.items():
            if key not in self.config:
                self.config[key] = value

    def generate_processes(self, config):
        # division config
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']
        division_config = dict(
            config.get('division', {}),
            daughter_path=daughter_path,
            agent_id=agent_id,
            compartment=self)

        growth_config = config.get('growth', {})
        growth_rate = config['growth_rate']
        growth_config['growth_rate'] = growth_rate

        inclusion_config = config.get('inclusion_body', {})

        return {
            'inclusion_body': InclusionBody(inclusion_config),
            'growth': Growth(growth_config),
            # 'front_growth': GrowthProtein(growth_config),
            # 'back_growth': GrowthProtein(growth_config),
            'division': MetaDivision(division_config)}

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        agents_path = config['agents_path']
        return {
            'inclusion_body': {
                'front': ('front',),
                'back': ('back',),
                'molecules': ('internal',),
                'global': boundary_path,
            },

            'growth': {
                'global': boundary_path
            },

            # 'front_growth': {
            #     'internal': ('front',),
            #     'global': boundary_path},
            # 'back_growth': {
            #     'internal': ('back',),
            #     'global': boundary_path},

            'division': {
                'global': boundary_path,
                'cells': agents_path
            },
        }


DEFAULT_CONFIG = {
    'inclusion_body': {
        'molecules_list': ['glucose'],
        'growth_rate': 1e-1,
    }
}


if __name__ == '__main__':
    out_dir = os.path.join(COMPARTMENT_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    agent_id = '0'
    parameters = copy.deepcopy(DEFAULT_CONFIG)
    parameters['agent_id'] = agent_id
    compartment = InclusionBodyGrowth(parameters)

    # settings for simulation and plot
    settings = {
        'outer_path': ('agents', agent_id),
        'return_raw_data': True,
        'timestep': 1,
        'total_time': 600}
    output_data = simulate_compartment_in_experiment(compartment, settings)

    plot_settings = {}
    plot_agents_multigen(output_data, plot_settings, out_dir)