
from vivarium_cell.experiments.control import control
from vivarium.core.composition import (
    compartment_hierarchy_experiment,
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



def run_experiment(out_dir):
    agent_id = '1'
    time_total = 600
    molecules = ['glucose']

    # initial state
    initial_state = {
        'agents': {
            agent_id: {
                'internal': {'A': 0},
                'external': {'A': 1},
                'global': {
                    'divide': False
                }
            },
        }
    }

    lattice_config = make_lattice_config(
        molecules=molecules,
        bounds=[30, 30],
        n_bins=[1, 1])

    # declare the hierarchy
    hierarchy = {
        'generators': [
            {
                'name': 'lattice',
                'type': Lattice,
                'config': lattice_config
            }
        ],
        'agents': {
            agent_id: {
                'generators': {
                    'type': InclusionBodyGrowth,
                    'config': {'agent_id': agent_id}
                },
            }
        }
    }

    # configure experiment
    experiment = compartment_hierarchy_experiment(
        hierarchy=hierarchy,
        initial_state=initial_state)

    # run simulation
    experiment.update(time_total)
    data = experiment.emitter.get_data()
    experiment.end()

    # multigen plot
    plot_settings = {}
    plot_agents_multigen(data, plot_settings, out_dir)

    # extract data for snapshots
    multibody_config = lattice_config['multibody']
    agents = {time: time_data['agents'] for time, time_data in data.items()}
    fields = {time: time_data['fields'] for time, time_data in data.items()}
    plot_data = {
        'agents': agents,
        'fields': fields,
        'config': multibody_config,
    }
    plot_config = {
        'out_dir': out_dir,
        # 'filename': agent_type + '_snapshots',
    }
    plot_snapshots(plot_data, plot_config)




experiments_library = {
    '1': {
        'name': 'inclusion_lattice',
        'function': run_experiment},
}

if __name__ == '__main__':
    control(
        experiments_library=experiments_library,
        out_dir=EXPERIMENT_OUT_DIR)
