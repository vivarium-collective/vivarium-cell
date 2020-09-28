from __future__ import absolute_import, division, print_function

import os
import copy

from vivarium.library.dict_utils import keys_list
from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process,
    PROCESS_OUT_DIR,
)
from vivarium.plots.simulation_output import plot_simulation_output
from vivarium.library.units import units

from vivarium_cell.data.nucleotides import nucleotides


NAME = 'rna_degradation'

def all_subkeys(d):
    subkeys = set([])
    for ase in d.keys():
        subkeys = subkeys.union(set(d[ase].keys()))
    return list(subkeys)

def kinetics(E, S, kcat, km):
    return kcat * E * S / (S + km)

DEFAULT_TRANSCRIPT_DEGRADATION_KM = 1e-23

TOY_CONFIG = {
    'sequences': {
        'oA': 'GCC',
        'oAZ': 'GCCGUGCAC',
        'oB': 'AGUUGA',
        'oBY': 'AGUUGACGG'
    },
    'catalytic_rates': {
        'endoRNAse': 0.1
    },
    'michaelis_constants': {
        'transcripts': {
            'endoRNAse': {
                'oA': DEFAULT_TRANSCRIPT_DEGRADATION_KM,
                'oAZ': DEFAULT_TRANSCRIPT_DEGRADATION_KM,
                'oB': DEFAULT_TRANSCRIPT_DEGRADATION_KM,
                'oBY': DEFAULT_TRANSCRIPT_DEGRADATION_KM,
            }
        }
    }
}


class RnaDegradation(Process):

    name = NAME
    defaults = {
        'sequences': {},
        'catalytic_rates': {
            'endoRNAse': 0.1},
        'michaelis_constants': {
            'transcripts': {
                'endoRNAse': {}
            }
        },
        'global_deriver_key': 'global_deriver',
        'time_step': 1.0,
    }

    def __init__(self, initial_parameters=None):
        if not initial_parameters:
            initial_parameters = {}

        super(RnaDegradation, self).__init__(initial_parameters)

        self.derive_defaults('sequences', 'transcript_order', keys_list)
        self.derive_defaults('catalytic_rates', 'protein_order', keys_list)

        self.sequences = self.parameters['sequences']
        self.catalytic_rates = self.parameters['catalytic_rates']
        self.michaelis_constants = self.parameters['michaelis_constants']
        self.transcript_order = self.parameters['transcript_order']
        self.protein_order = self.parameters['protein_order']
        self.molecule_order = list(nucleotides.values())
        self.molecule_order.extend(['ATP', 'ADP'])

        self.partial_transcripts = {
            transcript: 0
            for transcript in self.transcript_order}

        self.global_deriver_key = self.or_default(
            initial_parameters, 'global_deriver_key')


    def ports_schema(self):

        ports = [
            'transcripts',
            'proteins',
            'molecules',
            'global']
        schema = {port: {} for port in ports}

        # transcripts
        for state in self.transcript_order:
            schema['transcripts'][state] = {
                '_default': 0,
                '_emit': True}

        # proteins
        for state in self.protein_order:
            schema['proteins'][state] = {
                '_default': 0,
                '_emit': True}

        # molecules
        for state in self.molecule_order:
            schema['molecules'][state] = {
                '_default': 0,
                '_emit': True}

        # global
        schema['global']['mmol_to_counts'] = {
            '_default': 1.0 * units.L / units.mmol}

        return schema

    def derivers(self):
        return {
            self.global_deriver_key: {
                'deriver': 'globals_deriver',
                'port_mapping': {
                    'global': 'global'},
                'config': {
                    'width': 1.11}}}

    def next_update(self, timestep, states):
        transcripts = states['transcripts']
        proteins = states['proteins']
        molecules = states['molecules']
        mmol_to_counts = states['global']['mmol_to_counts']

        delta_transcripts = {
            transcript: 0
            for transcript in self.transcript_order}

        for protein, kcat in self.catalytic_rates.items():
            for transcript, km in self.michaelis_constants['transcripts'][protein].items():
                km *= units.mole / units.fL
                delta_transcripts[transcript] += kinetics(
                    proteins[protein] / mmol_to_counts,
                    transcripts[transcript] / mmol_to_counts,
                    kcat,
                    km)

        degradation_levels = {
            transcript: (
                level * mmol_to_counts * timestep).magnitude + self.partial_transcripts[transcript]
            for transcript, level in delta_transcripts.items()}

        transcript_counts = {
            transcript: -int(level)
            for transcript, level in degradation_levels.items()}

        self.partial_transcripts = {
            transcript: level - int(level)
            for transcript, level in degradation_levels.items()}

        delta_molecules = {
            molecule: 0
            for molecule in self.molecule_order}

        for transcript, count in transcript_counts.items():
            sequence = self.sequences[transcript]
            for base in sequence:
                delta_molecules[nucleotides[base]] -= count
                # ATP hydrolysis cost is 1 per nucleotide degraded
                delta_molecules['ATP'] -= count
                delta_molecules['ADP'] += count

        return {
            'transcripts': transcript_counts,
            'molecules': delta_molecules}


def test_rna_degradation(end_time=10):
    rna_degradation = RnaDegradation(TOY_CONFIG)

    # initial state
    proteins = {
        protein: 10
        for protein in rna_degradation.protein_order}
    molecules = {
        molecule: 10
        for molecule in rna_degradation.molecule_order}
    molecules['ATP'] = 100000
    transcripts = {
        transcript: 10
        for transcript in rna_degradation.transcript_order}

    settings = {
        'total_time': end_time,
        'initial_state': {
            'molecules': molecules,
            'proteins': proteins,
            'transcripts': transcripts}}

    return simulate_process(rna_degradation, settings)


if __name__ == '__main__':
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plot_settings = {}
    timeseries = test_rna_degradation(100)
    plot_simulation_output(timeseries, plot_settings, out_dir)
