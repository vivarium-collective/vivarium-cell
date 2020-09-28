'''
=========================
Colony Metrics Experiment
=========================
'''

from __future__ import absolute_import, division, print_function

import os
import random

import numpy as np
import pytest

from vivarium.core.composition import (
    EXPERIMENT_OUT_DIR,
    make_agents,
    assert_timeseries_close,
    load_timeseries,
    simulate_experiment,
)
from vivarium.plots.agents_multigen import plot_agents_multigen
from vivarium.core.emitter import (
    path_timeseries_from_data,
    path_timeseries_from_embedded_timeseries,
    timeseries_from_data,
)
from vivarium.core.experiment import Experiment

from vivarium_cell.composites.lattice import (
    Lattice,
    make_lattice_config,
)
from vivarium_cell.experiments.lattice_experiment import agents_library
from vivarium_cell.plots.multibody_physics import plot_snapshots
from vivarium_cell.plots.colonies import plot_colony_metrics
from vivarium.library.timeseries import (
    process_path_timeseries_for_csv,
    save_flat_timeseries,
)
from vivarium_cell.processes.multibody_physics import single_agent_config
from vivarium_cell.data import REFERENCE_DATA_DIR

NAME = 'colony_metrics'
OUT_DIR = os.path.join(EXPERIMENT_OUT_DIR, NAME)
DEFAULT_BOUNDS = [40, 40]
DEFAULT_EMIT_STEP = 30


def colony_metrics_experiment(config):
    '''Run an experiment to calculate colony metrics

    Arguments:
        config (dict): A dictionary of configuration options which can
            contain the following:

            * **n_agents** (:py:class:`int`): Number of agents to create
              initially
            * **emitter** (:py:class:`dict`): Emitter configuration
              dictionary. This gets passed directly to the
              :py:class:`vivarium.core.experiment.Experiment`
              constructor. Defaults to a timeseries emitter.
            * **environment** (:py:class:`dict`): Configuration to pass
              to the :py:class:`vivarium.composites.lattice.Lattice`
              constructor. Defaults to ``{}``.
            * **agent** (:py:class:`dict`): Dictionary with the
              following keys-value pairs:

                * **compartment**
                  (:py:class:`vivarium.core.experiment.Compartment`): A
                  compartment class to use for each agent.
                * **config** (:py:class:`dict`): Configuration to pass
                  the agent compartment constructor.

            * **locations** (:py:class:`list`): List of sublists. Each
              sublist should have 2 elements, the coordinates at which
              to place an agent. Coordinates range from 0 to 1 and
              represent fractions of environment bounds. Any agents for
              which no location is specified will be placed randomly per
              the default behavior of
              :py:func:`vivarium.processes.multibody.single_agent_config`.

    Returns:
        vivarium.core.experiment.Experiment: An initialized experiment
        object.
    '''
    # configure the experiment
    n_agents = config.get('n_agents')
    emitter = config.get('emitter', {'type': 'timeseries'})

    # make lattice environment
    environment = Lattice(config.get('environment', {}))
    network = environment.generate()
    processes = network['processes']
    topology = network['topology']

    # add the agents
    agent_ids = [str(agent_id) for agent_id in range(n_agents)]
    agent_config = config['agent']
    agent_compartment = agent_config['type']
    compartment_config = agent_config['config']
    agent = agent_compartment(compartment_config)
    agents = make_agents(agent_ids, agent, {})
    processes['agents'] = agents['processes']
    topology['agents'] = agents['topology']

    # initial agent state
    locations = config.get('locations')
    if locations is None:
        locations = [[0.5, 0.5]]
    agent_config_settings = [
        {
            'bounds': environment.config['multibody']['bounds'],
            'location': random.choice(locations) if len(locations) <= index else locations[index]
        }
        for index, agent_id in enumerate(agent_ids)
    ]

    initial_state = {
        'agents': {
            agent_id: single_agent_config(agent_config_setting)
            for agent_id, agent_config_setting
            in zip(agent_ids, agent_config_settings)
        }
    }
    initial_state.update(config.get('initial_state', {}))

    return Experiment({
        'processes': processes,
        'topology': topology,
        'emitter': emitter,
        'initial_state': initial_state,
    })


def get_lattice_with_metrics_config():
    config = {
        'environment': make_lattice_config(
            bounds=DEFAULT_BOUNDS,
        )
    }
    colony_metrics_config = {
        'colony_shape_deriver': {
            'alpha': 1 / 5,
        },
    }
    config['environment'].update(colony_metrics_config)
    return config


def run_experiment(
    runtime=2400, n_agents=2, start_locations=None, growth_rate=0.000275
):
    '''Run a Colony Metrics Experiment

    Arguments:
        runtime (int): Experiment duration
        n_agents (int): Number of agents to create at the start
        start_locations (list): List of initial agent coordinates. If
            you do not provide enough locations, the remaining locations
            will be random. Coordinates range from 0 to 1 and represent
            fractions of environment bounds.

    Returns:
        Simulation data as :term:`raw data`.
    '''
    agent_config = agents_library['growth_division_minimal']
    agent_config['config']['growth_rate_noise'] = 0
    agent_config['growth_rate'] = growth_rate

    experiment_config = get_lattice_with_metrics_config()
    experiment_config.update({
        'n_agents': n_agents,
        'agent': agent_config,
        'locations': start_locations,
    })
    experiment = colony_metrics_experiment(experiment_config)

    # simulate
    settings = {
        'emit_step': DEFAULT_EMIT_STEP,
        'total_time': runtime,
        'return_raw_data': True,
    }
    return simulate_experiment(experiment, settings), experiment_config


@pytest.mark.slow
def test_experiment(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    data, _ = run_experiment(
        start_locations=[[0.3, 0.3], [0.5, 0.5]],
        growth_rate=0.001,
    )
    path_ts = path_timeseries_from_data(data)
    filtered = {
        path: timeseries
        for path, timeseries in path_ts.items()
        # Angles are computed randomly by multibody physics
        if path[-1] != 'angle'
    }
    processed_for_csv = process_path_timeseries_for_csv(filtered)
    save_flat_timeseries(
        processed_for_csv,
        OUT_DIR,
        'test_output.csv'
    )
    test_output = load_timeseries(os.path.join(OUT_DIR, 'test_output.csv'))
    expected = load_timeseries(
        os.path.join(REFERENCE_DATA_DIR, NAME + '.csv'))
    assert_timeseries_close(
        test_output, expected,
        default_tolerance=(1 - 1e-5),
    )


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    data, experiment_config = run_experiment(
        start_locations=[[0.3, 0.3], [0.5, 0.5]],
    )

    # extract data
    multibody_config = experiment_config['environment']['multibody']
    agents = {time: time_data['agents'] for time, time_data in data.items()}
    fields = {time: time_data['fields'] for time, time_data in data.items()}

    # agents plot
    agents_settings = {
        'agents_key': 'agents'
    }
    plot_agents_multigen(data, agents_settings, OUT_DIR, 'agents')

    # snapshot plot
    snapshot_data = {
        'agents': agents,
        'fields': fields,
        'config': multibody_config,
    }
    snapshot_config = {
        'out_dir': OUT_DIR,
        'filename': 'agents_snapshots',
    }
    plot_snapshots(snapshot_data, snapshot_config)

    # Colony Metrics Plot
    embedded_ts = timeseries_from_data(data)
    colony_metrics_ts = embedded_ts['colony_global']
    colony_metrics_ts['time'] = embedded_ts['time']
    path_ts = path_timeseries_from_embedded_timeseries(
        colony_metrics_ts)
    fig = plot_colony_metrics(path_ts)
    fig.savefig(os.path.join(OUT_DIR, 'colonies'))


if __name__ == '__main__':
    main()
