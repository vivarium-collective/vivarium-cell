"""
==========================
Inclusion Body Experiments
==========================
"""

from vivarium.core.control import Control
from vivarium.core.composition import (
    compose_experiment,
    GENERATORS_KEY,
    EXPERIMENT_OUT_DIR,
)

# composites
from vivarium_cell.composites.lattice import (
    Lattice,
    make_lattice_config,
)
from vivarium_cell.composites.inclusion_body_growth import InclusionBodyGrowth

# plots
from vivarium.plots.agents_multigen import plot_agents_multigen
from vivarium_cell.plots.multibody_physics import (
    plot_snapshots,
    plot_tags
)


def experiment_config():
    pass

lattice_config = make_lattice_config(
    jitter_force=1e-4,
    bounds=[30, 30],
    n_bins=[10, 10])



def run_experiment():
    agent_id = '1'
    time_total = 5000

    inclusion_config = {
        'agent_id': agent_id,
        'inclusion_process': {
            'damage_rate': 1e-3,
        }
    }

    # initial state
    compartment = InclusionBodyGrowth(inclusion_config)
    compartment_state = compartment.initial_state({
        'inclusion_body': {
            'front': 0.9,
            'back': 0.1,
        }})
    initial_state = {
        'agents': {
            agent_id: compartment_state}}

    # lattice_config = make_lattice_config(
    #     jitter_force=1e-4,
    #     bounds=[30, 30],
    #     n_bins=[10, 10])

    # declare the hierarchy
    hierarchy = {
        GENERATORS_KEY:{
            'type': Lattice,
            'config': lattice_config},
        'agents': {
            agent_id: {
                GENERATORS_KEY: {
                    'type': InclusionBodyGrowth,
                    'config': inclusion_config}}}}

    # configure experiment
    experiment = compose_experiment(
        hierarchy=hierarchy,
        initial_state=initial_state)

    # run simulation
    experiment.update(time_total)
    data = experiment.emitter.get_data()
    experiment.end()

    return data


def inclusion_plots_suite(data=None, out_dir=EXPERIMENT_OUT_DIR):
    n_snapshots = 6
    tagged_molecules = [
        ('inclusion_body',),
    ]

    # multigen plot
    plot_settings = {}
    plot_agents_multigen(data, plot_settings, out_dir)

    # extract data for snapshots
    multibody_config = lattice_config['multibody']
    agents = {time: time_data['agents'] for time, time_data in data.items()}

    # snapshots plot
    plot_data = {
        'agents': agents,
        'config': multibody_config,
    }
    plot_config = {
        'n_snapshots': n_snapshots,
        'out_dir': out_dir}
    plot_snapshots(plot_data, plot_config)

    # tags plot
    plot_config = {
        'tagged_molecules': tagged_molecules,
        'n_snapshots': n_snapshots,
        'convert_to_concs': False,
        'out_dir': out_dir}
    plot_tags(plot_data, plot_config)



# libraries for control
experiments_library = {
    '1': {
        'name': 'inclusion_lattice',
        'experiment': run_experiment},
}
plots_library = {
    '1': inclusion_plots_suite
}
workflow_library = {
    '1': {
        'name': 'inclusion_body_experiment',
        'experiment': '1',
        'plots': ['1'],
    }
}

if __name__ == '__main__':
    Control(
        experiments=experiments_library,
        plots=plots_library,
        workflows=workflow_library,
        )
