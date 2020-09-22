'''
========================
Stochastic Transcription
========================
'''

import os
import copy
import numpy as np
import logging as log
from arrow import StochasticSystem

from vivarium.library.units import units
from vivarium.library.dict_utils import deep_merge, keys_list
from vivarium.core.experiment import pp, pf
from vivarium.core.process import Process
from vivarium.core.composition import process_in_experiment
from vivarium_cell.states.chromosome import Chromosome, Rnap, Promoter, frequencies, add_merge, toy_chromosome_config
from vivarium_cell.library.polymerize import Elongation, build_stoichiometry, template_products
from vivarium_cell.data.nucleotides import nucleotides

def choose_element(elements):
    if elements:
        choice = np.random.choice(len(elements), 1)
        return list(elements)[int(choice)]

#: Variable name for unbound RNA polymerase
UNBOUND_RNAP_KEY = 'RNA Polymerase'

monomer_ids = list(nucleotides.values())

#: The default configuration parameters for :py:class:`Transcription`
class Transcription(Process):

    name = 'transcription'
    defaults = {
        'promoter_affinities': {},
        'transcription_factors': [],
        'sequence': '',
        'templates': {},
        'genes': {},
        'elongation_rate': 1.0,
        'polymerase_occlusion': 5,
        'symbol_to_monomer': nucleotides,
        'monomer_ids': monomer_ids,
        'concentrations_deriver_key': 'transcription_concentrations',
        'initial_domains': {
            0: {
                'id': 0,
                'lead': 0,
                'lag': 0,
                'children': []}},
        'molecule_ids': monomer_ids,
        'time_step': 1.0,
    }

    def __init__(self, initial_parameters=None):
        '''A stochastic transcription model

        .. WARNING:: Vivarium's knowledge base uses the gene name to
            name the protein. This means that for a gene acrA that
            codes for protein AcrA, you must refer to the gene,
            transcript, and protein each as acrA.

        :term:`Ports`:

        * **chromosome**: The linked :term:`store` should hold a
          representation of the chromosome in the form returned by
          :py:meth:`vivarium_cell.states.chromosome.Chromosome.to_dict`.
        * **molecules**: Expects variables with the names in the
          *molecule_ids* configuration. These are the monomers consumed
          by transcription.
        * **factors**: Expects variables for each transcription factor's
          concentration.
        * **transcripts**: The linked store should store the
          concentrations of the transcripts.
        * **proteins**: The linked store should hold the concentrations
          of the transcription factors and the RNA polymerase.

        Arguments:
            initial_parameters: The following configuration options may
                be provided:

                * **promoter_affinities** (:py:class:`dict`): Maps from
                  binding state tuples to the binding affinity of RNA
                  polymerase and the promoter when the promoter is at
                  that binding state. The binding state of a promoter is
                  which (if any) transcription factors are bound to the
                  promoter. Such a binding state can be represented by a
                  binding state tuple, which is a :py:class:`tuple`
                  whose first element is the name of the promoter. All
                  bound transcription factors are listed as subsequent
                  elements. If no transcription factors are bound, the
                  sole subsequent element is ``None``.

                  .. todo:: What is the significance of the order in the
                      binding state tuple?

                  .. todo:: What are the units of the affinities?

                * **transcription_factors** (:py:class:`list`): A list
                  of all modeled transcription factors.
                * **sequence**: The DNA sequence that includes all the
                  genes whose transcription is being modeled.
                * **templates** (:py:class:`dict`): Maps from the name
                  of an operon to that operon's :term:`template
                  specification`.
                * **genes** (:py:class:`dict`): Maps from operon name to
                  a list of the names of the genes in that operon.
                * **elongation_rate** (:py:class:`float`): The
                  elongation rate of the RNA polymerase.
                * **polymerase_occlusion** (:py:class:`int`): The number
                  of base pairs behind the polymerase where another
                  polymerase is occluded and so cannot bind.
                * **symbol_to_monomer** (:py:class:`dict`): Maps from
                  the symbols used to represent monomers in the RNA
                  sequence to the name of the free monomer. This should
                  generally be
                  :py:data:`cell.data.nucleotides.nucleotides`.
                * **monomer_ids** (:py:class:`list`): A list of the
                  names of the free monomers consumed by transcription.
                  This can generally be computed as:

                  >>> from vivarium_cell.data.nucleotides import nucleotides
                  >>> monomer_ids = nucleotides.values()
                  >>> print(list(monomer_ids))
                  ['ATP', 'GTP', 'UTP', 'CTP']

                  Note that we only included the ``list()``
                  transformation to make the output prettier. The
                  ``dict_values`` object returned by the ``.values()``
                  call is sufficiently list-like for use here.
                * **molecule_ids** (:py:class:`list`): A list of all the
                  molecules needed by the :term:`process`. This will
                  generally be the same as *monomer_ids*.

        Example configuring the process (uses
        :py:func:`vivarium.library.pretty.format_dict`):

        >>> import random
        >>>
        >>> import numpy as np
        >>>
        >>> from vivarium_cell.states.chromosome import (
        ...     toy_chromosome_config,
        ...     Chromosome,
        ... )
        >>> from vivarium_cell.data.nucleotides import nucleotides
        >>> # format_dict lets us print dictionaries prettily
        >>> from vivarium.library.pretty import format_dict
        >>>
        >>> random.seed(0)  # Needed because process is stochastic
        >>> np.random.seed(0)
        >>> # We will use the toy chromosome from toy_chromosome_config
        >>> print(toy_chromosome_config)
        {'sequence': 'ATACGGCACGTGACCGTCAACTTA', 'genes': {'oA': ['eA'], 'oAZ': ['eA', 'eZ'], 'oB': ['eB'], 'oBY': ['eB', 'eY']}, 'promoter_order': ['pA', 'pB'], 'promoters': {'pA': {'id': 'pA', 'position': 3, 'direction': 1, 'sites': [{'position': 0, 'length': 3, 'thresholds': {'tfA': <Quantity(0.3, 'millimolar')>}}], 'terminators': [{'position': 6, 'strength': 0.5, 'products': ['oA']}, {'position': 12, 'strength': 1.0, 'products': ['oAZ']}]}, 'pB': {'id': 'pB', 'position': -3, 'direction': -1, 'sites': [{'position': 0, 'length': 3, 'thresholds': {'tfB': <Quantity(0.5, 'millimolar')>}}], 'terminators': [{'position': -9, 'strength': 0.5, 'products': ['oB']}, {'position': -12, 'strength': 1.0, 'products': ['oBY']}]}}, 'promoter_affinities': {('pA', None): 1.0, ('pA', 'tfA'): 10.0, ('pB', None): 1.0, ('pB', 'tfB'): 10.0}, 'domains': {0: {'id': 0, 'lead': 0, 'lag': 0, 'children': []}}, 'rnaps': {}}
        >>> monomer_ids = list(nucleotides.values())
        >>> configuration = {
        ...     'promoter_affinities': {
        ...         ('pA', None): 1.0,
        ...         ('pA', 'tfA'): 10.0,
        ...         ('pB', None): 1.0,
        ...         ('pB', 'tfB'): 10.0},
        ...     'transcription_factors': ['tfA', 'tfB'],
        ...     'sequence': toy_chromosome_config['sequence'],
        ...     'templates': toy_chromosome_config['promoters'],
        ...     'genes': toy_chromosome_config['genes'],
        ...     'elongation_rate': 10.0,
        ...     'polymerase_occlusion': 5,
        ...     'symbol_to_monomer': nucleotides,
        ...     'monomer_ids': monomer_ids,
        ...     'molecule_ids': monomer_ids,
        ... }
        >>> # At this point we haven't used the toy chromosome yet
        >>> # because it will be specified in the chromosome port.
        >>> # Notice that the parameters are specific to the chromosome.
        >>> transcription_process = Transcription(configuration)
        >>> # Now we need to initialize the simulation stores
        >>> state = {
        ...     'chromosome': toy_chromosome_config,
        ...     'molecules': {
        ...         nucleotide: 10
        ...         for nucleotide in monomer_ids
        ...     },
        ...     'proteins': {UNBOUND_RNAP_KEY: 10},
        ...     'factors': {'tfA': 0.2 * units.mM, 'tfB': 0.7 * units.mM},
        ... }
        >>> update = transcription_process.next_update(1.0, state)
        >>> print(update['chromosome'])
        {'rnaps': {'_add': [{'path': (2,), 'state': <class 'vivarium_cell.states.chromosome.Rnap'>: {'id': 2, 'template': 'pA', 'template_index': 0, 'terminator': 1, 'domain': 0, 'state': 'polymerizing', 'position': 7}}, {'path': (3,), 'state': <class 'vivarium_cell.states.chromosome.Rnap'>: {'id': 3, 'template': 'pB', 'template_index': 1, 'terminator': 0, 'domain': 0, 'state': 'occluding', 'position': 3}}, {'path': (4,), 'state': <class 'vivarium_cell.states.chromosome.Rnap'>: {'id': 4, 'template': 'pA', 'template_index': 0, 'terminator': 0, 'domain': 0, 'state': 'occluding', 'position': 0}}], '_delete': []}, 'rnap_id': 4, 'domains': {0: <class 'vivarium_cell.states.chromosome.Domain'>: {'id': 0, 'lead': 0, 'lag': 0, 'children': []}}, 'root_domain': 0}
        '''

        if not initial_parameters:
            initial_parameters = {}

        log.debug('inital transcription parameters: {}'.format(initial_parameters))

        super(Transcription, self).__init__(initial_parameters)

        self.derive_defaults('templates', 'promoter_order', keys_list)
        self.derive_defaults('templates', 'transcript_ids', template_products)

        self.sequence = self.parameters['sequence']
        self.templates = self.parameters['templates']
        self.genes = self.parameters['genes']
        empty_chromosome = Chromosome({
            'sequence': self.sequence,
            'promoters': self.templates,
            'genes': self.genes})
        self.sequences = empty_chromosome.sequences()
        self.symbol_to_monomer = self.parameters['symbol_to_monomer']

        log.debug('chromosome sequence: {}'.format(self.sequence))

        self.promoter_affinities = self.parameters['promoter_affinities']
        self.promoter_order = self.parameters['promoter_order']
        self.promoter_count = len(self.promoter_order)

        self.transcription_factors = self.parameters['transcription_factors']
        self.molecule_ids = self.parameters['molecule_ids']
        self.molecule_ids.extend(['ATP', 'ADP'])
        self.monomer_ids = self.parameters['monomer_ids']
        self.transcript_ids = self.parameters['transcript_ids']
        self.elongation = 0
        self.elongation_rate = self.parameters['elongation_rate']
        self.polymerase_occlusion = self.parameters['polymerase_occlusion']

        self.stoichiometry = build_stoichiometry(self.promoter_count)
        self.initiation = StochasticSystem(self.stoichiometry, random_seed=np.random.randint(2**31))

        self.protein_ids = [UNBOUND_RNAP_KEY] + self.transcription_factors

        self.initial_domains = self.parameters['initial_domains']
        self.concentrations_deriver_key = self.parameters['concentrations_deriver_key']

        self.chromosome_ports = ['rnaps', 'rnap_id', 'domains', 'root_domain']

        log.debug('final transcription parameters: {}'.format(self.parameters))

    def build_affinity_vector(self, promoters, factors):
        vector = np.zeros(len(self.promoter_order), dtype=np.float64)
        for index, promoter_key in enumerate(self.promoter_order):
            promoter = promoters[promoter_key]
            binding = promoter.binding_state(factors)
            affinity = self.promoter_affinities.get(binding, 0.0)
            # print('promoter state - {}: {}'.format(binding, affinity))
            vector[index] = affinity
        return vector

    def chromosome_config(self, chromosome_states):
        return dict(
            chromosome_states,
            sequence=self.sequence,
            promoters=self.templates,
            promoter_order=self.promoter_order,
            genes=self.genes)

    def ports_schema(self):
        schema = {}

        schema['chromosome'] = {
            'rnap_id': {
                '_default': 1,
                '_updater': 'set'},
            'root_domain': {
                '_default': 0,
                '_updater': 'set'},
            'domains': {
                '*': {
                    'id': {
                        '_default': 1,
                        '_updater': 'set'},
                    'lead': {
                        '_default': 0,
                        '_updater': 'set'},
                    'lag': {
                        '_default': 0,
                        '_updater': 'set'},
                    'children': {
                        '_default': [],
                        '_updater': 'set'}}},
            'rnaps': {
                '*': {
                    'id': {
                        '_default': -1,
                        '_updater': 'set'},
                    'domain': {
                        '_default': 0,
                        '_updater': 'set'},
                    'state': {
                        '_default': None,
                        '_updater': 'set',
                        '_emit': True},
                    'position': {
                        '_default': 0,
                        '_updater': 'set',
                        '_emit': True},
                    'template': {
                        '_default': None,
                        '_updater': 'set',
                        '_emit': True},
                    'template_index': {
                        '_default': 0,
                        '_updater': 'set',
                        '_emit': True},
                    'terminator': {
                        '_default': 0,
                        '_updater': 'set',
                        '_emit': True}}}}

        initial_domains = {
            id: {
                'id': {
                    '_default': id,
                    '_updater': 'set'},
                'lead': {
                    '_default': 0,
                    '_updater': 'set'},
                'lag': {
                    '_default': 0,
                    '_updater': 'set'},
                'children': {
                    '_default': [],
                    '_updater': 'set'}}
            for id, domain in self.initial_domains.items()}
        schema['chromosome']['domains'].update(initial_domains)

        schema['molecules'] = {
            molecule: {
                '_default': 0,
                '_divider': 'split',
                '_emit': True}
            for molecule in self.molecule_ids}

        schema['factors'] = {
            factor: {
                '_default': 0.0,
                '_divider': 'split'}
            for factor in self.transcription_factors}

        schema['transcripts'] = {
            protein: {
                '_default': 0,
                '_divider': 'split',
                '_emit': True}
            for protein in self.transcript_ids}

        schema['proteins'] = {
            protein: {
                '_default': 0,
                '_divider': 'split',
                '_emit': True}
            for protein in self.protein_ids}

        schema['global'] = {}

        return schema

    def derivers(self):
        return {
            self.concentrations_deriver_key: {
                'deriver': 'concentrations_deriver',
                'port_mapping': {
                    'global': 'global',
                    'counts': 'proteins',
                    'concentrations': 'factors'},
                'config': {
                    'concentration_keys': self.transcription_factors}}}

    def next_update(self, timestep, states):
        chromosome_state = states['chromosome']
        # chromosome_state['rnaps'] = list(chromosome_state['rnaps'].values())
        original_rnap_keys = [
            rnap['id'] for rnap in chromosome_state['rnaps'].values()]
        chromosome = Chromosome(
            self.chromosome_config(chromosome_state))

        molecules = states['molecules']
        proteins = states['proteins']
        factors = states['factors'] # as concentrations

        promoter_rnaps = chromosome.promoter_rnaps()
        promoter_domains = chromosome.promoter_domains()

        # Find out how many promoters are currently blocked by a
        # newly initiated or occluding rnap
        promoter_count = len(chromosome.promoter_order)
        blocked_promoters = np.zeros(promoter_count, dtype=np.int64)
        open_domains = {}
        bound_domains = {}
        for promoter_index, promoter_key in enumerate(chromosome.promoter_order):
            domains = []
            for rnap in promoter_rnaps.get(promoter_key, {}).values():
                if rnap.is_occluding():
                    domains.append(rnap.domain)
                    blocked_promoters[promoter_index] += 1

            bound_domains[promoter_key] = set(domains)
            open_domains[promoter_key] = promoter_domains[promoter_key] - bound_domains[promoter_key]

        blocked_promoters = np.array(blocked_promoters)

        # Make the state for a gillespie simulation out of total number of each
        # promoter by copy number not blocked by initiated rnap,
        # concatenated with the number of each promoter that is bound by rnap.
        # These are the two states for each promoter the simulation
        # will operate on, essentially going back and forth between
        # bound and unbound states.
        copy_numbers = chromosome.promoter_copy_numbers()
        original_unbound_rnaps = proteins[UNBOUND_RNAP_KEY]
        monomer_limits = {
            monomer: molecules[monomer]
            for monomer in self.monomer_ids}
        unbound_rnaps = original_unbound_rnaps

        time = 0
        now = 0
        elongation = Elongation(
            self.sequences,
            chromosome.promoters,
            monomer_limits,
            self.symbol_to_monomer,
            self.elongation)

        initiation_affinity = self.build_affinity_vector(chromosome.promoters, factors)

        while time < timestep:
            # build the state vector for the gillespie simulation
            substrate = np.concatenate([
                copy_numbers - blocked_promoters,
                blocked_promoters,
                [unbound_rnaps]])

            log.debug('transcription substrate: {}'.format(substrate))
            log.debug('blocked promoters: {}'.format(blocked_promoters))

            # find number of monomers until next terminator
            distance = 1 / self.elongation_rate # chromosome.terminator_distance()

            # find interval of time that elongates to the point of the next terminator
            interval = min(distance, timestep - time)

            if interval == distance:
                # perform the elongation until the next event
                terminations, monomer_limits, chromosome.rnaps = elongation.step(
                    interval,
                    monomer_limits,
                    chromosome.rnaps)
                unbound_rnaps += terminations
            else:
                elongation.store_partial(interval)
                terminations = 0

            log.debug('time: {} --- interval: {}'.format(time, interval))
            log.debug('monomer limits: {}'.format(monomer_limits))
            log.debug('terminations: {}'.format(terminations))

            # run simulation for interval of time to next terminator
            result = self.initiation.evolve(
                interval,
                substrate,
                initiation_affinity)

            log.debug('result: {}'.format(result))

            # perform binding
            for now, event in zip(result['time'], result['events']):
                # RNAP has bound the promoter
                promoter_key = chromosome.promoter_order[event]
                promoter = chromosome.promoters[promoter_key]
                domains = open_domains[promoter_key]
                domain = choose_element(domains)

                blocked_promoters[event] += 1
                bound_domains[promoter_key].add(domain)
                open_domains[promoter_key].remove(domain)

                # create a new bound RNAP and add it to the chromosome.
                new_rnap = chromosome.bind_rnap(event, domain)
                new_rnap.start_polymerizing()

                log.debug('newly bound RNAP: {}'.format(new_rnap))

                unbound_rnaps -= 1

            # deal with occluding rnap
            for rnap in chromosome.rnaps.values():
                if rnap.is_unoccluding(self.polymerase_occlusion):
                    log.debug('RNAP unoccluding: {}'.format(rnap))

                    blocked_promoters[rnap.template_index] -= 1
                    bound_domains[rnap.template].remove(rnap.domain)
                    open_domains[rnap.template].add(rnap.domain)
                    rnap.unocclude()
                log.debug('rnap: {}'.format(rnap))

            log.debug('complete: {}'.format(elongation.complete_polymers))

            time += interval

        # track how far elongation proceeded to start from next iteration
        self.elongation = elongation.elongation - int(elongation.elongation)

        proteins = {
            UNBOUND_RNAP_KEY: unbound_rnaps - original_unbound_rnaps}

        molecules = {
            key: count * -1
            for key, count in elongation.monomers.items()}

        # 1 ATP hydrolysis cost per nucleotide elongation
        molecules['ATP'] = 0
        molecules['ADP'] = 0
        for count in elongation.monomers.values():
            molecules['ATP'] -= count
            molecules['ADP'] += count

        chromosome_dict = chromosome.to_dict()
        rnaps = chromosome_dict['rnaps']

        original = set(original_rnap_keys)
        current = set(rnaps.keys())
        bound_rnaps = current - original
        completed_rnaps = original - current
        continuing_rnaps = original - completed_rnaps

        rnap_updates = {
            rnap_id: rnaps[rnap_id]
            for rnap_id in continuing_rnaps}

        add_rnaps = [
            {'path': (bound,), 'state': rnaps[bound]}
            for bound in bound_rnaps]

        delete_rnaps = [
            (completed,)
            for completed in completed_rnaps]

        rnap_updates['_add'] = add_rnaps
        rnap_updates['_delete'] = delete_rnaps
        chromosome_dict['rnaps'] = rnap_updates

        update = {
            'chromosome': {
                key: chromosome_dict[key]
                for key in self.chromosome_ports},
            'proteins': proteins,
            'molecules': molecules,
            'transcripts': elongation.complete_polymers}

        log.debug('molecules update: {}'.format(update['molecules']))

        return update


def test_transcription():
    parameters = {
        'sequence': toy_chromosome_config['sequence'],
        'templates': toy_chromosome_config['promoters'],
        'genes': toy_chromosome_config['genes'],
        'promoter_affinities': toy_chromosome_config['promoter_affinities'],
        'transcription_factors': ['tfA', 'tfB'],
        'elongation_rate': 10.0}

    chromosome = Chromosome(toy_chromosome_config)
    transcription = Transcription(parameters)

    initial_molecules = {
                nucleotide: 10
                for nucleotide in transcription.monomer_ids}
    initial_molecules['ATP'] = 100000

    experiment = process_in_experiment(transcription, {
        'initial_state': {
            'chromosome': chromosome.to_dict(),
            'molecules': initial_molecules,
            'proteins': {UNBOUND_RNAP_KEY: 10},
            'factors': {'tfA': 0.2, 'tfB': 0.7}}})

    pp(experiment.state.get_value())
    experiment.update(10.0)
    pp(experiment.state.get_value())

    print('complete!')



if __name__ == '__main__':
    test_transcription()
