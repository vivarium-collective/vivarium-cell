from __future__ import absolute_import, division, print_function

import os
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

from vivarium.core.process import Generator
from vivarium.core.composition import (
    COMPARTMENT_OUT_DIR,
    simulate_compartment_in_experiment,
    plot_simulation_output,
)
from vivarium.library.make_network import save_network
from vivarium.library.units import units

# processes
from vivarium.processes.tree_mass import TreeMass
from cell.data.amino_acids import amino_acids
from cell.processes.transcription import Transcription, UNBOUND_RNAP_KEY
from cell.processes.translation import Translation, UNBOUND_RIBOSOME_KEY
from cell.processes.degradation import RnaDegradation
from cell.processes.complexation import Complexation
from cell.processes.division_volume import DivisionVolume
from cell.data.amino_acids import amino_acids
from cell.data.nucleotides import nucleotides
from cell.states.chromosome import toy_chromosome_config


NAME = 'gene_expression'

class GeneExpression(Generator):

    defaults = {
        'global_path': ('global',),
        'initial_mass': 1339.0 * units.fg,
        'time_step': 1.0,
        'transcription': {},
        'translation': {},
        'degradation': {},
        'complexation': {},
    }

    def __init__(self, config):
        super(GeneExpression, self).__init__(config)

    def generate_processes(self, config):
        transcription_config = config['transcription']
        translation_config = config['translation']
        degradation_config = config['degradation']
        complexation_config = config['complexation']

        # update timestep
        transcription_config.update({'time_step': config['time_step']})
        translation_config.update({'time_step': config['time_step']})
        degradation_config.update({'time_step': config['time_step']})
        complexation_config.update({'time_step': config['time_step']})

        # make the processes
        transcription = Transcription(transcription_config)
        translation = Translation(translation_config)
        degradation = RnaDegradation(degradation_config)
        complexation = Complexation(complexation_config)
        mass_deriver = TreeMass(config.get('mass_deriver', {
            'initial_mass': config['initial_mass']}))
        division = DivisionVolume(config)

        return {
            'mass_deriver': mass_deriver,
            'transcription': transcription,
            'translation': translation,
            'degradation': degradation,
            'complexation': complexation,
            'division': division}

    def generate_topology(self, config):
        global_path = config['global_path']

        return {
            'mass_deriver': {
                'global': global_path},

            'transcription': {
                'chromosome': ('chromosome',),
                'molecules': ('molecules',),
                'proteins': ('proteins',),
                'transcripts': ('transcripts',),
                'factors': ('concentrations',),
                'global': global_path},

            'translation': {
                'ribosomes': ('ribosomes',),
                'molecules': ('molecules',),
                'transcripts': ('transcripts',),
                'proteins': ('proteins',),
                'concentrations': ('concentrations',),
                'global': global_path},

            'degradation': {
                'transcripts': ('transcripts',),
                'proteins': ('proteins',),
                'molecules': ('molecules',),
                'global': global_path},

            'complexation': {
                'monomers': ('proteins',),
                'complexes': ('proteins',),
                'global': global_path},

            'division': {
                'global': global_path}}



# analysis
def gene_network_plot(data, out_dir, filename='gene_network'):
    '''
    Make a gene network plot from configuration data

    - data (dict):
        {
        'operons': operons, the "genes" in a chromosome config with {operon: [genes list]}
        'templates': promoters, the "promoters" in a chromosome config with {promoter: {sites: [], thresholds: []}}
        'complexes': complexes, stoichiometry from a complexation process.
        }

    '''
    node_size = 400
    node_distance = 30
    edge_weight = 1
    iterations = 1000

    operon_suffix = '_o'
    tf_suffix = '_tf'
    promoter_suffix = '_p'
    gene_suffix = '_g'
    complex_suffix = '_cxn'

    # plotting parameters
    color_legend = {
        'operon': [x/255 for x in [199,164,53]],
        'gene': [x / 255 for x in [181,99,206]],
        'promoter': [x / 255 for x in [110,196,86]],
        'transcription_factor': [x / 255 for x in [222,85,80]],
        'complex': [x / 255 for x in [153, 204, 255]],
    }

    # get data
    operons = data.get('operons', {})
    templates = data.get('templates', {})
    complexes = data.get('complexes', {})

    # make graph from templates
    G = nx.Graph()

    # initialize node lists for graphing
    operon_nodes = set()
    gene_nodes = set()
    promoter_nodes = set()
    tf_nodes = set()
    complex_nodes = set()

    edges = []

    # add operon --> gene connections
    for operon, genes in operons.items():
        operon_name = operon + operon_suffix
        gene_names = [gene + gene_suffix for gene in genes]

        operon_nodes.add(operon_name)
        gene_nodes.update(gene_names)

        G.add_node(operon_name)
        G.add_nodes_from(gene_names)

        # set node attributes
        operon_attrs = {operon_name: {'type': 'operon'}}
        gene_attrs = {gene: {'type': 'gene'} for gene in gene_names}
        nx.set_node_attributes(G, operon_attrs)
        nx.set_node_attributes(G, gene_attrs)

        # add operon --> gene edge
        for gene_name in gene_names:
            edge = (operon_name, gene_name)
            edges.append(edge)
            G.add_edge(operon_name, gene_name)

    # add transcription factor --> promoter --> operon connections
    for promoter, specs in templates.items():
        promoter_name = promoter + promoter_suffix

        promoter_nodes.add(promoter_name)
        G.add_node(promoter_name)

        # set node attributes
        promoter_attrs = {promoter_name: {'type': 'promoter'}}
        nx.set_node_attributes(G, promoter_attrs)

        # get sites and terminators
        promoter_sites = specs['sites']
        terminators = specs['terminators']

        # add transcription factors
        for site in promoter_sites:
            thresholds = site['thresholds']
            for threshold in thresholds:
                tf_name = threshold + tf_suffix

                tf_nodes.add(tf_name)
                G.add_node(tf_name)

                # set node attributes
                tf_attrs = {tf_name: {'type': 'transcription_factor'}}
                nx.set_node_attributes(G, tf_attrs)

                # connect transcription_factor --> promoter
                edge = (tf_name, promoter_name)
                edges.append(edge)
                G.add_edge(tf_name, promoter_name)

        # add gene products
        for site in terminators:
            products = site['products']
            for product in products:
                operon_name = product + operon_suffix

                operon_nodes.add(operon_name)
                G.add_node(operon_name)

                # set node attributes
                operon_attrs = {operon_name: {'type': 'operon'}}
                nx.set_node_attributes(G, operon_attrs)

                # connect promoter --> operon
                edge = (promoter_name, operon_name)
                edges.append(edge)
                G.add_edge(promoter_name, operon_name)


    ## add gene product --> complex
    # first get the sets of complexes and reactants.
    complex_set = set()
    monomer_set = set()
    for complex, stoichiometry in complexes.items():
        complex_list = [mol_id for mol_id, coeff in stoichiometry.items() if coeff>0]
        reactant_list = [mol_id for mol_id, coeff in stoichiometry.items() if coeff<0]

        complex_set.update(complex_list)
        monomer_set.update(reactant_list)

    # complexes are removed from reactants
    for complex in complex_set:
        if complex in monomer_set:
            monomer_set.remove(complex)

    # add nodes and edges
    for complex, stoichiometry in complexes.items():
        complexes = [mol_id for mol_id, coeff in stoichiometry.items() if coeff>0]
        reactants = {mol_id: coeff for mol_id, coeff in stoichiometry.items() if coeff<0}

        assert len(complexes) == 1, 'too many complexes'
        complex = complexes[0]
        complex_name = complex + complex_suffix

        complex_nodes.add(complex_name)
        G.add_node(complex_name)

        # set node attributes
        complex_attrs = {complex_name: {'type': 'complex'}}
        nx.set_node_attributes(G, complex_attrs)

        # make edge
        for reactant, coeff in reactants.items():

            # is this reactant a monomer or a complex?
            if reactant in monomer_set:
                # TODO -- check that it is actually included in the genes?
                reactant_name = reactant + gene_suffix
            elif reactant in complex_set:
                reactant_name = reactant + complex_suffix

            # connect reactant --> complex
            edge = (reactant_name, complex_name)
            edges.append(edge)
            G.add_edge(reactant_name, complex_name)

    # add edges between proteins/complexes and transcription factors that share the same name
    tf_names = [node.replace(tf_suffix, '') for node in tf_nodes]
    for node in gene_nodes:
        gene_name = node.replace(gene_suffix, '')
        if gene_name in tf_names:
            tf_node = gene_name + tf_suffix
            edge = (node, tf_node)
            edges.append(edge)
            G.add_edge(node, tf_node)

    for node in complex_nodes:
        complex_name = node.replace(complex_suffix, '')
        if complex_name in tf_names:
            tf_node = complex_name + tf_suffix
            edge = (node, tf_node)
            edges.append(edge)
            G.add_edge(node, tf_node)

    # separate out the disconnected graphs
    subgraphs = sorted(nx.connected_components(G), key = len, reverse=True)

    # make node labels by removing suffix
    node_labels = {}
    o_labels = {node: node.replace(operon_suffix,'') for node in operon_nodes}
    g_labels = {node: node.replace(gene_suffix, '') for node in gene_nodes}
    p_labels = {node: node.replace(promoter_suffix, '') for node in promoter_nodes}
    tf_labels = {node: node.replace(tf_suffix, '') for node in tf_nodes}
    cxn_labels = {node: node.replace(complex_suffix, '') for node in complex_nodes}
    node_labels.update(o_labels)
    node_labels.update(g_labels)
    node_labels.update(p_labels)
    node_labels.update(tf_labels)
    node_labels.update(cxn_labels)

    # save network for use in gephi
    gephi_nodes = {}
    gephi_edges = [[node1, node2, edge_weight] for (node1, node2) in edges]
    for node_id in operon_nodes:
        gephi_nodes[node_id] = {
            'label': o_labels[node_id],
            'type': 'operon',
            'size': 1}
    for node_id in gene_nodes:
        gephi_nodes[node_id] = {
            'label': g_labels[node_id],
            'type': 'gene',
            'size': 1}
    for node_id in promoter_nodes:
        gephi_nodes[node_id] = {
            'label': p_labels[node_id],
            'type': 'promoter',
            'size': 1}
    for node_id in tf_nodes:
        gephi_nodes[node_id] = {
            'label': tf_labels[node_id],
            'type': 'transcription factor',
            'size': 1}
    for node_id in complex_nodes:
        gephi_nodes[node_id] = {
            'label': cxn_labels[node_id],
            'type': 'complex',
            'size': 1}
    save_network(gephi_nodes, gephi_edges, out_dir)


    # plot
    n_rows = len(list(subgraphs))
    n_cols = 1
    fig = plt.figure(3, figsize=(6*n_cols, 6*n_rows))
    grid = plt.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2)

    for idx, subgraph_nodes in enumerate(subgraphs):
        ax = fig.add_subplot(grid[idx, 0])

        subgraph = G.subgraph(list(subgraph_nodes))
        subgraph_node_labels = {node: node_labels[node] for node in subgraph_nodes}

        # get positions
        dist = node_distance / (len(subgraph)**0.5)
        # pos = nx.spring_layout(subgraph, k=dist, iterations=iterations)
        pos = nx.spring_layout(subgraph,
                               k=dist,
                               scale=20,
                               iterations=iterations)

        color_map = []
        for node in subgraph:
            type = subgraph.nodes[node]['type']
            node_color = color_legend[type]
            color_map.append(node_color)

        nx.draw(subgraph, pos, node_size=node_size, node_color=color_map)

        # edges
        # colors = list(range(1,len(edges)+1))
        nx.draw_networkx_edges(subgraph, pos,
                               # edge_color=colors,
                               width=1.5)

        nx.draw_networkx_labels(subgraph, pos,
                                labels=subgraph_node_labels,
                                font_size=8)

        if idx == 0:
            # make legend
            legend_elements = [Patch(facecolor=color, label=name) for name, color in color_legend.items()]
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.5, 1.0))

    # save figure
    fig_path = os.path.join(out_dir, filename)
    plt.figure(3, figsize=(12, 12))
    plt.axis('off')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def plot_gene_expression_output(timeseries, config, out_dir='out'):

    name = config.get('name', 'gene_expression')
    ports = config.get('ports', {})
    molecules = timeseries[ports['molecules']]
    transcripts = timeseries[ports['transcripts']]
    proteins = timeseries[ports['proteins']]
    time = timeseries['time']

    # make figure and plot
    n_cols = 1
    n_rows = 5
    plt.figure(figsize=(n_cols * 6, n_rows * 1.5))

    # define subplots
    ax1 = plt.subplot(n_rows, n_cols, 1)
    ax2 = plt.subplot(n_rows, n_cols, 2)
    ax3 = plt.subplot(n_rows, n_cols, 3)
    ax4 = plt.subplot(n_rows, n_cols, 4)
    ax5 = plt.subplot(n_rows, n_cols, 5)

    polymerase_ids = [
        UNBOUND_RNAP_KEY,
        UNBOUND_RIBOSOME_KEY]
    amino_acid_ids = list(amino_acids.values())
    nucleotide_ids = list(nucleotides.values())

    # plot polymerases
    for poly_id in polymerase_ids:
        ax1.plot(time, proteins[poly_id], label=poly_id)
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax1.title.set_text('polymerases')

    # plot nucleotides
    for nuc_id in nucleotide_ids:
        ax2.plot(time, molecules[nuc_id], label=nuc_id)
    ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax2.title.set_text('nucleotides')

    # plot molecules
    for aa_id in amino_acid_ids:
        ax3.plot(time, molecules[aa_id], label=aa_id)
    ax3.legend(loc='center left', bbox_to_anchor=(2.0, 0.5))
    ax3.title.set_text('amino acids')

    # plot transcripts
    for transcript_id, series in transcripts.items():
        ax4.plot(time, series, label=transcript_id)
    ax4.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax4.title.set_text('transcripts')

    # plot proteins
    for protein_id in sorted(proteins.keys()):
        if protein_id != UNBOUND_RIBOSOME_KEY:
            ax5.plot(time, proteins[protein_id], label=protein_id)
    ax5.legend(loc='center left', bbox_to_anchor=(1.5, 0.5))
    ax5.title.set_text('proteins')

    # adjust axes
    for axis in [ax1, ax2, ax3, ax4, ax5]:
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    ax5.set_xlabel('time (s)', fontsize=12)

    # save figure
    fig_path = os.path.join(out_dir, name)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig(fig_path, bbox_inches='tight')


# test
def run_gene_expression(total_time=10, out_dir='out'):
    timeseries = test_gene_expression(total_time)
    plot_settings = {
        'name': 'gene_expression',
        'ports': {
            'transcripts': 'transcripts',
            'molecules': 'molecules',
            'proteins': 'proteins'}}
    plot_gene_expression_output(timeseries, plot_settings, out_dir)

    sim_plot_settings = {'max_rows': 25}
    plot_simulation_output(timeseries, sim_plot_settings, out_dir)

def test_gene_expression(total_time=10):
    # load the compartment
    compartment_config = {
        'external_path': ('external',),
        'global_path': ('global',),
        'agents_path': ('..', '..', 'cells',),
        'transcription': {
            'sequence': toy_chromosome_config['sequence'],
            'templates': toy_chromosome_config['promoters'],
            'genes': toy_chromosome_config['genes'],
            'promoter_affinities': toy_chromosome_config['promoter_affinities'],
            'transcription_factors': ['tfA', 'tfB'],
            'elongation_rate': 10.0},
        # 'complexation': {
        #     'monomer_ids': [],
        #     'complex_ids': [],
        #     'stoichiometry': {}}
    }
    compartment = GeneExpression(compartment_config)

    molecules = {
        nt: 1000
        for nt in nucleotides.values()}
    molecules.update({
        aa: 1000
        for aa in amino_acids.values()})

    proteins = {
        polymerase: 100
        for polymerase in [
                UNBOUND_RNAP_KEY,
                UNBOUND_RIBOSOME_KEY]}

    proteins.update({
        factor: 1
        for factor in [
                'tfA',
                'tfB']})

    # simulate
    settings = {
        'timestep': 1,
        'total_time': total_time,
        'initial_state': {
            'proteins': proteins,
            'molecules': molecules}}
    return simulate_compartment_in_experiment(compartment, settings)


if __name__ == '__main__':
    out_dir = os.path.join(COMPARTMENT_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_gene_expression(600, out_dir)
