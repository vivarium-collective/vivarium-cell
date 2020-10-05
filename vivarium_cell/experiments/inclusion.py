
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
    time_total = 100
    molecules = ['glucose']
    n_snapshots = 6
    tagged_molecules = [
        ('front', 'inclusion_body'),
        ('back', 'inclusion_body'),
    ]

    # initial state
    compartment = InclusionBodyGrowth({'agent_id': agent_id})
    compartment_state = compartment.initial_state({
        'internal': {'glucose': 1.0}})
    initial_state = {
        'agents': {
            agent_id: compartment_state}}

    lattice_config = make_lattice_config(
        molecules=molecules,
        bounds=[30, 30],
        n_bins=[10, 10])

    # declare the hierarchy
    hierarchy = {
        'generators': [
            {
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

    # snapshots plot
    plot_data = {
        'agents': agents,
        'fields': fields,
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



experiments_library = {
    '1': {
        'name': 'inclusion_lattice',
        'experiment': run_experiment},
}

if __name__ == '__main__':
    control(
        experiments_library=experiments_library,
        out_dir=EXPERIMENT_OUT_DIR)
