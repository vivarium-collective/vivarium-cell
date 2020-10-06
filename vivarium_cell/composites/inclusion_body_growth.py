from __future__ import absolute_import, division, print_function

import os
import copy

from vivarium.library.units import units
from vivarium.library.dict_utils import deep_merge
from vivarium.core.process import Generator
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    COMPARTMENT_OUT_DIR,
)
from vivarium.plots.agents_multigen import plot_agents_multigen

# processes
from vivarium.processes.meta_division import MetaDivision
from vivarium_cell.processes.inclusion_body import InclusionBody
from vivarium_cell.processes.growth_rate import GrowthRate
from vivarium_cell.processes.divide_condition import DivideCondition
from vivarium_cell.processes.derive_globals import DeriveGlobals

from vivarium_cell.experiments.control import control


NAME = 'inclusion_body_growth'


class InclusionBodyGrowth(Generator):

    defaults = {
        'inclusion_process': {},
        'growth_rate': {
            'growth_rate': 0.001},  # fast growth
        'divide_condition': {
            'threshold': 3000 * units.fg},
        'mass': {},
        'boundary_path': ('boundary',),
        'agents_path': ('..', '..', 'agents',),
        'daughter_path': tuple()}

    def __init__(self, config):
        super(InclusionBodyGrowth, self).__init__(config)

    def initial_state(self, config=None):
        if config is None:
            config = {}
        # get the processes
        network = self.generate()
        processes = network['processes']
        topology = network['topology']

        initial_state = {}
        for name, process in processes.items():
            if name in ['inclusion_process', 'growth_rate']:
                process_state = process.initial_state()

                # replace port name with store name
                # TODO -- find a way to build this into the generator...
                process_topology = topology[name]
                replace_port_id = {}
                for port_id, state in process_state.items():
                    store_id = process_topology[port_id][0]
                    if port_id is not store_id:
                        replace_port_id[port_id] = store_id
                for port_id, store_id in replace_port_id.items():
                    process_state[store_id] = process_state[port_id]
                    del process_state[port_id]

                initial_state = deep_merge(initial_state, process_state)
        deep_merge(initial_state, config)
        return initial_state

    def generate_processes(self, config):
        # division config
        daughter_path = config['daughter_path']
        agent_id = config['agent_id']
        division_config = dict(
            config.get('division', {}),
            daughter_path=daughter_path,
            agent_id=agent_id,
            compartment=self)

        return {
            'inclusion_process': InclusionBody(config['inclusion_process']),
            'growth_rate': GrowthRate(config['growth_rate']),
            'globals_deriver': DeriveGlobals({}),
            'divide_condition': DivideCondition(config['divide_condition']),
            'division': MetaDivision(division_config)
        }

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        agents_path = config['agents_path']
        return {
            'inclusion_process': {
                'inclusion_mass': ('inclusion_body',),
                'molecules': ('internal',),
                'global': boundary_path,
            },
            'growth_rate': {
                'global': boundary_path
            },
            'globals_deriver': {
                'global': boundary_path
            },
            'divide_condition': {
                'variable': boundary_path + ('mass',),
                'divide': boundary_path + ('divide',),
            },
            'division': {
                'global': boundary_path,
                'agents': agents_path
            },
        }


DEFAULT_CONFIG = {
    'inclusion_process': {
        'molecules_list': ['glucose'],
        'growth_rate': 1e-1,
    }
}

def test_inclusion_body(total_time=1000):
    agent_id = '0'
    parameters = copy.deepcopy(DEFAULT_CONFIG)
    parameters['agent_id'] = agent_id
    compartment = InclusionBodyGrowth(parameters)

    initial_state = compartment.initial_state({
        'internal': {
            'glucose': 1.0}})

    # settings for simulation and plot
    settings = {
        'initial_state': {'agents': {agent_id: initial_state}},
        'outer_path': ('agents', agent_id),
        'return_raw_data': True,
        'timestep': 1,
        'total_time': total_time}
    return simulate_compartment_in_experiment(compartment, settings)

def run_compartment(out_dir):
    output_data = test_inclusion_body(
        total_time=4000)
    plot_settings = {}
    plot_agents_multigen(output_data, plot_settings, out_dir)


experiments_library = {
    '1': {
        'name': 'inclusion_body_growth',
        'experiment': run_compartment,
    }
}


if __name__ == '__main__':
    control(
        experiments_library=experiments_library,
        out_dir=COMPARTMENT_OUT_DIR)
