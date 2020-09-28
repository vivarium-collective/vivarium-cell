from __future__ import absolute_import, division, print_function

import os

from vivarium.core.process import Generator
from vivarium.core.composition import (
    COMPARTMENT_OUT_DIR,
    simulate_compartment_in_experiment,
)
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.library.units import units

# processes
from vivarium.processes.tree_mass import TreeMass

from vivarium_cell.plots.gene_expression import plot_gene_expression_output
from vivarium_cell.processes.transcription import Transcription, UNBOUND_RNAP_KEY
from vivarium_cell.processes.translation import Translation, UNBOUND_RIBOSOME_KEY
from vivarium_cell.processes.degradation import RnaDegradation
from vivarium_cell.processes.complexation import Complexation
from vivarium_cell.processes.division_volume import DivisionVolume
from vivarium_cell.data.amino_acids import amino_acids
from vivarium_cell.data.nucleotides import nucleotides
from vivarium_cell.states.chromosome import toy_chromosome_config


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
