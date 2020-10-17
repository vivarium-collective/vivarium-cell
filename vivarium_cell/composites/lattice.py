from __future__ import absolute_import, division, print_function

import copy
import os

from vivarium.core.process import Generator
from vivarium.core.composition import (
    compartment_in_experiment,
    COMPARTMENT_OUT_DIR,
)
from vivarium.library.dict_utils import deep_merge

# processes
from vivarium_cell.processes.multibody_physics import (
    Multibody,
    agent_body_config,
)
from vivarium_cell.processes.diffusion_field import (
    DiffusionField,
    get_gaussian_config,
)
from vivarium_cell.processes.derive_colony_shape import ColonyShapeDeriver

# plots
from vivarium_cell.plots.multibody_physics import plot_snapshots


NAME = 'lattice'


# make a configuration dictionary for the Lattice compartment
def make_lattice_config(
        time_step=None,
        jitter_force=None,
        bounds=None,
        n_bins=None,
        depth=None,
        concentrations=None,
        molecules=None,
        diffusion=None,
        keep_fields_emit=None,
        set_config=None,
        parallel=None,
):
    config = {'multibody': {}, 'diffusion': {}}

    if time_step:
        config['multibody']['time_step'] = time_step
        config['diffusion']['time_step'] = time_step
    if bounds:
        config['multibody']['bounds'] = bounds
        config['diffusion']['bounds'] = bounds
        config['diffusion']['n_bins'] = bounds
    if n_bins:
        config['diffusion']['n_bins'] = n_bins
    if jitter_force:
        config['multibody']['jitter_force'] = jitter_force
    if depth:
        config['diffusion']['depth'] = depth
    if diffusion:
        config['diffusion']['diffusion'] = diffusion
    if concentrations:
        config['diffusion']['gradient'] = {
            'type': 'uniform',
            'molecules': concentrations}
        molecules = list(concentrations.keys())
        config['diffusion']['molecules'] = molecules
    elif molecules:
        # molecules are a list, assume uniform concentrations of 1
        config['diffusion']['molecules'] = molecules
    if keep_fields_emit:
        # by default no fields are emitted
        config['diffusion']['_schema'] = {
            'fields': {
                field_id: {
                    '_emit': False}
                for field_id in molecules
                if field_id not in keep_fields_emit}}
    if parallel:
        config['diffusion']['_parallel'] = True
        config['multibody']['_parallel'] = True
    if set_config:
        config = deep_merge(config, set_config)

    return config



class Lattice(Generator):
    """
    Lattice:  A two-dimensional lattice environmental model with multibody physics and diffusing molecular fields.
    """

    name = 'lattice_environment'
    defaults = {
        # To exclude a process, from the compartment, set its
        # configuration dictionary to None, e.g. colony_mass_deriver
        'multibody': {
            'bounds': [10, 10],
            'size': [10, 10],
        },
        'diffusion': {
            'molecules': ['glc'],
            'n_bins': [10, 10],
            'size': [10, 10],
            'depth': 3000.0,
            'diffusion': 1e-2,
        },
        'colony_shape_deriver': None,
        '_schema': {},
    }

    def __init__(self, config=None):
        super(Lattice, self).__init__(config)

    def generate_processes(self, config):
        processes = {
            'multibody': Multibody(config['multibody']),
            'diffusion': DiffusionField(config['diffusion'])
        }
        colony_shape_config = config['colony_shape_deriver']
        if colony_shape_config is not None:
            processes['colony_shape_deriver'] = ColonyShapeDeriver(
                colony_shape_config)
        return processes

    def generate_topology(self, config):
        topology = {
            'multibody': {
                'agents': ('agents',),
            },
            'diffusion': {
                'agents': ('agents',),
                'fields': ('fields',),
                'dimensions': ('dimensions',),
            },
            'colony_shape_deriver': {
                'colony_global': ('colony_global',),
                'agents': ('agents',),
            }
        }
        return {
            process: process_topology
            for process, process_topology in topology.items()
            if config[process] is not None
        }


def test_lattice(
        config=None,
        n_agents=1,
        end_time=10
):
    if config is None:
        config = make_lattice_config()
    # configure the compartment
    compartment = Lattice(config)

    # set initial agent state
    if n_agents:
        agent_ids = [str(agent_id) for agent_id in range(n_agents)]
        body_config = {'agent_ids': agent_ids}
        if 'multibody' in config and 'bounds' in config['multibody']:
            body_config.update({'bounds': config['multibody']['bounds']})
        initial_agents_state = agent_body_config(body_config)
        initial_state = {'agents': initial_agents_state}

    # configure experiment
    experiment_settings = {
        'compartment': config,
        'initial_state': initial_state}
    experiment = compartment_in_experiment(
        compartment,
        experiment_settings)

    # run experiment
    timestep = 1
    time = 0
    while time < end_time:
        experiment.update(timestep)
        time += timestep
    data = experiment.emitter.get_data()

    # assert that the agent remains in the simulation until the end
    assert len(data[end_time]['agents']) == n_agents
    return data


def main():
    out_dir = os.path.join(COMPARTMENT_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    bounds = [25, 25]
    config = make_lattice_config(
        bounds=bounds,
    )
    data = test_lattice(
        config=config,
        n_agents=1,
        end_time=40)

    # make snapshot plot
    agents = {time: time_data['agents'] for time, time_data in data.items()}
    fields = {time: time_data['fields'] for time, time_data in data.items()}
    plot_data = {
        'agents': agents,
        'fields': fields,
        'config': {'bounds': bounds}}
    plot_config = {
        'out_dir': out_dir,
        'filename': 'snapshots'}
    plot_snapshots(plot_data, plot_config)


if __name__ == '__main__':
    main()
