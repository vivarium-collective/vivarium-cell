"""
========================
Inclusion body composite
========================
"""

import os

from vivarium.library.units import units
from vivarium.library.dict_utils import deep_merge
from vivarium.core.process import Generator
from vivarium.core.composition import (
    simulate_compartment_in_experiment,
    COMPOSITE_OUT_DIR,
)
from vivarium.plots.agents_multigen import plot_agents_multigen
from vivarium.plots.topology import plot_compartment_topology

# processes
from vivarium.processes.meta_division import MetaDivision
from vivarium_cell.processes.inclusion_body import InclusionBody
from vivarium_cell.processes.growth_rate import GrowthRate
from vivarium.processes.divide_condition import DivideCondition
from vivarium_cell.processes.derive_globals import DeriveGlobals



NAME = 'inclusion_body_growth'


class InclusionBodyGrowth(Generator):

    defaults = {
        'inclusion_process': {},
        'growth_rate': {},
        'divide_condition': {
            'threshold': 3000 * units.fg},
        'mass': {},
        'boundary_path': ('boundary',),
        'agents_path': ('..', '..', 'agents',),
        'daughter_path': tuple(),
        'initial_state_config': {
            'inclusion_process': {
                'initial_mass': 10},
            'growth_rate': {
                'initial_mass': 1200}}}

    def __init__(self, config):
        super(InclusionBodyGrowth, self).__init__(config)

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
                'front': ('front',),
                'back': ('back',),
                'inclusion_body': ('inclusion_body',),
                'molecules': ('molecules',),
            },
            'growth_rate': {
                'molecules': ('molecules',),
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


    def initial_state(self, config=None):
        # TODO -- find a way to build this into the generator...

        if config is None:
            config = {}
        initial_state_config = self.config['initial_state_config']
        config = deep_merge(config, initial_state_config)

        # get the processes
        network = self.generate()
        processes = network['processes']
        topology = network['topology']

        initial_state = {}
        for name, process in processes.items():
            if name in ['inclusion_process', 'growth_rate']:
                process_state = process.initial_state(config.get(name, {}))

                # replace port name with store name
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

        return initial_state


def test_inclusion_body(total_time=1000):
    agent_id = '0'
    parameters = {
        'agent_id': agent_id,
        'inclusion_process': {
            'damage_rate': 1e-4,  # rapid damage
        },
        'growth_rate': {
            'growth_rate': 0.001  # fast growth
        },
    }
    compartment = InclusionBodyGrowth(parameters)

    initial_state = compartment.initial_state()

    # settings for simulation and plot
    settings = {
        'initial_state': {'agents': {agent_id: initial_state}},
        'outer_path': ('agents', agent_id),
        'return_raw_data': True,
        'timestep': 1,
        'total_time': total_time}
    return simulate_compartment_in_experiment(compartment, settings)

def run_compartment(out_dir='out'):
    data = test_inclusion_body(total_time=4000)
    plot_settings = {}
    plot_agents_multigen(data, plot_settings, out_dir)

def plot_inclusion_topology(out_dir='out'):
    # make a topology network plot
    plot_compartment_topology(
        compartment=InclusionBodyGrowth({'agent_id': '1'}),
        settings={},
        out_dir=out_dir)


if __name__ == '__main__':
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plot_inclusion_topology(out_dir=out_dir)
    run_compartment(out_dir=out_dir)
