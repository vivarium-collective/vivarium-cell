from __future__ import absolute_import, division, print_function

import os
import uuid
import math
import random

from vivarium.core.experiment import (
    generate_state,
    Experiment
)
from vivarium.core.composition import (
    make_agents,
    simulate_experiment,
)
from vivarium.plots.agents_multigen import plot_agents_multigen

# composites
from vivarium_cell.composites.lattice import Lattice
from vivarium_cell.composites.growth_division_minimal import GrowthDivisionMinimal

# processes
from vivarium.library.units import units
from vivarium_cell.processes.derive_globals import volume_from_length
from vivarium_cell.processes.multibody_physics import (
    volume_from_length, DEFAULT_BOUNDS, PI
)
from vivarium_cell.plots.multibody_physics import plot_snapshots


def mother_machine_experiment(config):
    # configure the experiment
    agent_ids = config.get('agent_ids', [])
    emitter = config.get('emitter', {'type': 'timeseries'})

    # get the environment
    environment = Lattice(config.get('environment', {}))
    network = environment.generate({})
    processes = network['processes']
    topology = network['topology']

    # get the agents
    growth_division = GrowthDivisionMinimal(config.get('growth_division', {}))
    agents = make_agents(agent_ids, growth_division, config.get('growth_division', {}))
    processes['agents'] = agents['processes']
    topology['agents'] = agents['topology']

    return Experiment({
        'processes': processes,
        'topology': topology,
        'initial_state': config.get('initial_state', {}),
        'emitter': emitter,
    })



# configurations
def mother_machine_body_config(config):
    '''
    Gets initial agent body locations given the mother machine set up
    '''
    # cell dimensions
    width = 1
    length = 2
    volume = volume_from_length(length, width)

    agent_ids = config['agent_ids']
    bounds = config.get('bounds', DEFAULT_BOUNDS)
    channel_space = config.get('channel_space', 1)
    n_agents = len(agent_ids)

    # possible locations, shuffled for index-in
    n_spaces = math.floor(bounds[0]/channel_space)
    assert n_agents < n_spaces, 'more agents than mother machine spaces'

    possible_locations = [
        [x*channel_space - channel_space/2, 0.01]
        for x in range(1, n_spaces)]
    random.shuffle(possible_locations)

    initial_agents = {
        agent_id: {
            'boundary': {
                'location': possible_locations[index],
                'angle': PI/2,
                'volume': volume,
                'length': length,
                'width': width}}
        for index, agent_id in enumerate(agent_ids)}
    return initial_agents


def get_mother_machine_config():
    bounds = [20, 20]
    n_bins = [10, 10]
    channel_height = 0.7 * bounds[1]
    channel_space = 1.5
    n_agents = 3

    agent_ids = [str(agent_id) for agent_id in range(n_agents)]

    ## growth division agent
    growth_division_config = {
        'agents_path': ('..', '..', 'agents'),
        'global_path': ('global',),
        'growth_rate': 0.006,  # fast!
        'division_volume': 2.6}

    ## environment
    # multibody
    multibody_config = {
        'animate': False,
        'mother_machine': {
            'channel_height': channel_height,
            'channel_space': channel_space},
        'jitter_force': 1e-3,
        'bounds': bounds}

    body_config = {
        'bounds': bounds,
        'channel_height': channel_height,
        'channel_space': channel_space,
        'agent_ids': agent_ids}
    initial_agents = mother_machine_body_config(body_config)

    # diffusion
    diffusion_config = {
        'molecules': ['glc'],
        'gradient': {
            'type': 'gaussian',
            'molecules': {
                'glc':{
                    'center': [0.5, 0.5],
                    'deviation': 3},
            }},
        'diffusion': 5e-3,
        'n_bins': n_bins,
        'size': bounds}

    return {
        'initial_state': {
            'agents': initial_agents},
        'agent_ids': agent_ids,
        'growth_division': growth_division_config,
        'environment': {
            'multibody': multibody_config,
            'diffusion': diffusion_config}}


def run_mother_machine(time=5, out_dir='out'):
    mother_machine_config = get_mother_machine_config()
    experiment = mother_machine_experiment(mother_machine_config)

    # simulate
    settings = {
        'emit_step': 5,
        'total_time': time,
        'return_raw_data': True}
    data = simulate_experiment(experiment, settings)

    # agents plot
    plot_settings = {
        'agents_key': 'agents'}
    plot_agents_multigen(data, plot_settings, out_dir)

    # snapshot plot
    multibody_config = mother_machine_config['environment']['multibody']
    agents = {time: time_data['agents'] for time, time_data in data.items()}
    fields = {time: time_data['fields'] for time, time_data in data.items()}
    snapshot_data = {
        'agents': agents,
        'fields': fields,
        'config': multibody_config}
    plot_config = {
        'out_dir': out_dir,
        'filename': 'snapshots'}

    plot_snapshots(snapshot_data, plot_config)


if __name__ == '__main__':
    out_dir = os.path.join('out', 'experiments', 'mother_machine')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_mother_machine(400, out_dir)
