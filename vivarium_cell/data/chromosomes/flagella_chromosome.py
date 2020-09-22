import os

from vivarium.library.units import units
from vivarium.library.fasta import read_sequence
from vivarium_cell.library.polymerize import generate_template
from vivarium_cell.data.knowledge_base import KnowledgeBase
from vivarium_cell.states.chromosome import Chromosome


ECOLI_GENOME_FILE_NAME = 'Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna.chromosome.Chromosome.fa'
ECOLI_GENOME_PATH = os.path.abspath(
    os.path.join(
        __file__,
        "../../flat/{}".format(ECOLI_GENOME_FILE_NAME)
    )
)


class FlagellaChromosome(object):
    ecoli_sequence = None
    knowledge_base = None

    def __init__(self, parameters={}):
        if self.knowledge_base is None:
            self.knowledge_base = KnowledgeBase()
        if self.ecoli_sequence is None:
            self.ecoli_sequence = read_sequence(ECOLI_GENOME_PATH)

        self.factor_thresholds = {
            ('flhDp', 'CRP'): 1e-05 * units.mM,
            ('fliLp1', 'flhDC'): 1e-06 * units.mM,
            ('fliLp1', 'fliA'): 1.3e-05 * units.mM,
            ('fliEp1', 'flhDC'): 4e-06 * units.mM,
            ('fliEp1', 'fliA'): 1.1e-05 * units.mM,
            ('fliFp1', 'flhDC'): 7e-06 * units.mM,
            ('fliFp1', 'fliA'): 1e-05 * units.mM,
            ('flgBp', 'flhDC'): 1e-05 * units.mM,
            ('flgBp', 'fliA'): 8e-06 * units.mM,
            ('flgAp', 'flhDC'): 1.3e-05 * units.mM,
            ('flgAp', 'fliA'): 6e-06 * units.mM,
            ('flhBp', 'flhDC'): 1.5e-05 * units.mM,
            ('flhBp', 'fliA'): 5e-06 * units.mM,
            ('fliAp1', 'flhDC'): 1.7e-05 * units.mM,
            ('fliAp1', 'fliA'): 4e-06 * units.mM,
            ('flgEp', 'flhDC'): 1.9e-05 * units.mM,
            ('flgEp', 'fliA'): 3e-06 * units.mM,
            ('fliDp', 'flhDC'): 1.9e-05 * units.mM,
            ('fliDp', 'fliA'): 3e-06 * units.mM,
            ('flgKp', 'flhDC'): 2.1e-05 * units.mM,
            ('flgKp', 'fliA'): 1e-06 * units.mM,
            ('fliCp', 'fliA'): 5e-06 * units.mM,
            ('tarp', 'fliA'): 7e-06 * units.mM,
            ('motAp', 'fliA'): 9e-06 * units.mM,
            ('flgMp', 'fliA'): 1.1e-06 * units.mM}

        self.factor_thresholds.update(parameters.get('thresholds', {}))

        self.chromosome_config = {
            'sequence': self.ecoli_sequence,
            'genes': {
                'flhDC': ['flhD', 'flhC'],
                'fliL': ['fliL', 'fliM', 'fliN', 'fliO', 'fliP', 'fliQ', 'fliR'],
                'fliE': ['fliE'],
                'fliF': ['fliF', 'fliG', 'fliH', 'fliI', 'fliJ', 'fliK'],
                'flgA': ['flgA', 'flgM', 'flgN'],
                'flgM': ['flgM', 'flgN'],
                'flgE': ['flgE'],
                'flgB': ['flgB', 'flgC', 'flgD', 'flgE', 'flgF', 'flgG', 'flgH', 'flgI', 'flgJ'],
                'flhB': ['flhB', 'flhA', 'flhE'],
                'fliA': ['fliA', 'fliZ'], # ignore 'tcyJ' for now
                'fliD': ['fliD', 'fliS', 'fliT'],
                'flgK': ['flgK', 'flgL'],
                'fliC': ['fliC'],
                'tar': ['tar', 'tap', 'cheR', 'cheB', 'cheY', 'cheZ'],
                'motA': ['motA', 'motB', 'cheA', 'cheW']},
            'promoters': {
                'flhDp': {
                    'id': 'flhDp',
                    'position': 1978197,
                    'direction': -1,
                    'sites': [{'thresholds': {'CRP': 1e-05}}],
                    'terminators': [
                        {
                            'position': 1977266,
                            'strength': 1.0,
                            'products': ['flhDC']}]},
                'fliLp1': {
                    'id': 'fliLp1',
                    'position': 2019618,
                    'direction': 1,
                    'sites': [
                        {'thresholds': {'flhDC': 4e-06}},
                        {'thresholds': {'fliA': 1.3e-05}}],
                    'terminators': [
                        {
                            'position': 2023678,
                            'strength': 1.0,
                            'products': ['fliL']}]},
                'fliEp1': {
                    'id': 'fliEp1',
                    'position': 2013014,
                    'direction': -1,
                    'sites': [
                        {'thresholds': {'flhDC': 5e-06}},
                        {'thresholds': {'fliA': 1.1e-05}}],
                    'terminators': [
                        {
                            'position': 2012700,
                            'strength': 1.0,
                            'products': ['fliE']}]},
                'fliFp1': {
                    'id': 'fliFp1',
                    'position': 2013229,
                    'direction': 1,
                    'sites': [
                        {'thresholds': {'flhDC': 7e-06}},
                        {'thresholds': {'fliA': 1e-05}}],
                    'terminators': [
                        {
                            'position': 2019513,
                            'strength': 1.0,
                            'products': ['fliF']}]},
                'flgBp': {
                    'id': 'flgBp',
                    'position': 1130863,
                    'direction': -1,
                    'sites': [
                        {'thresholds': {'flhDC': 1e-05}},
                        {'thresholds': {'fliA': 8e-06}}],
                    'terminators': [
                        {
                            'position': 1129414,
                            'strength': 1.0,
                            'products': ['flgA']}]},
                'flgAp': {
                    'id': 'flgAp',
                    'position': 1131018,
                    'direction': 1,
                    'sites': [
                        {'thresholds': {'flhDC': 1.3e-05}},
                        {'thresholds': {'fliA': 6e-06}}],
                    'terminators': [
                        {
                            'position': 1138312,
                            'strength': 1.0,
                            'products': ['flgB']}]},
                'flhBp': {
                    'id': 'flhBp',
                    'position': 1966191,
                    'direction': -1,
                    'sites': [
                        {'thresholds': {'flhDC': 1.5e-05}},
                        {'thresholds': {'fliA': 5e-06}}],
                    'terminators': [
                        {
                            'position': 1962580,
                            'strength': 1.0,
                            'products': ['flhB']}]},
                'fliAp1': {
                    'id': 'fliAp1',
                    'position': 2001789,
                    'direction': -1,
                    'sites': [
                        {'thresholds': {'flhDC': 1.7e-05}},
                        {'thresholds': {'fliA': 4e-06}}],
                    'terminators': [
                        {
                            'position': 1999585,
                            'strength': 1.0,
                            'products': ['fliA']}]},
                'flgEp': {
                    'id': 'flgEp',
                    'position': 1132574,
                    'direction': 1,
                    'sites': [
                        {'thresholds': {'flhDC': 1.9e-05}},
                        {'thresholds': {'fliA': 3e-06}}],
                    'terminators': [
                        {
                            'position': 1133782,
                            'strength': 1.0,
                            'products': ['flgE']}]},
                'fliDp': {
                    'id': 'fliDp',
                    'position': 2003872,
                    'direction': 1,
                    'sites': [
                        {'thresholds': {'flhDC': 1.9e-05}},
                        {'thresholds': {'fliA': 3e-06}}],
                    'terminators': [
                        {
                            'position': 2006078,
                            'strength': 1.0,
                            'products': ['fliD']}]},
                'flgKp': {
                    'id': 'flgKp',
                    'position': 1138378,
                    'direction': 1,
                    'sites': [
                        {'thresholds': {'flhDC': 2.1e-05}},
                        {'thresholds': {'fliA': 1e-06}}],
                    'terminators': [
                        {
                            'position': 1140986,
                            'strength': 1.0,
                            'products': ['flgK']}]},
                'fliCp': {
                    'id': 'fliCp',
                    'position': 2002110,
                    'direction': 1,
                    'sites': [
                        {'thresholds': {'fliA': 5e-06}}],
                        # {'thresholds': {'GadE': 0.55, 'H-NS': 0.6}}],
                    'terminators': [
                        {
                            'position': 2003606,
                            'strength': 1.0,
                            'products': ['fliC']}]},
                'tarp': {
                    'id': 'tarp',
                    'position': 1972691,
                    'direction': -1,
                    'sites': [{'thresholds': {'fliA': 7e-06}}],
                        # {'thresholds': {'Fnr': 1e-5}}
                    'terminators': [
                        {
                            'position': 1971030,
                            'strength': 1.0,
                            'products': ['tar']}]},
                'motAp': {
                    'id': 'motAp',
                    'position': 1977139,
                    'direction': -1,
                    'sites': [{'thresholds': {'fliA': 9e-06}}],
                        # {'thresholds': {'CpxR': 1e-5}}],
                    'terminators': [
                        {
                            'position': 1976252,
                            'strength': 1.0,
                            'products': ['motA']}]},
                'flgMp': {
                    'id': 'flgMp',
                    'position': 1130128,
                    'direction': -1,
                    'sites': [{'thresholds': {'fliA': 1.1e-05}}],
                        # {'thresholds': {'CsgD': 0.1}}
                    'terminators': [
                        {
                            'position': 1129414,
                            'strength': 1.0,
                            'products': ['flgM']}]}},
            'domains': {
                0: {
                    'id': 0,
                    'lead': 0,
                    'lag': 0,
                    'children': []}},
            'rnaps': {}}

        # build chromosome and apply thresholds
        self.chromosome = Chromosome(self.chromosome_config)
        self.chromosome.apply_thresholds(self.factor_thresholds)
        self.chromosome_config['promoters'] = {
            key: promoter.to_dict()
            for key, promoter in self.chromosome.promoters.items()}

        self.promoters = [
            'flhDp', 'fliLp1', 'fliEp1', 'fliFp1', 'flgAp', 'flgBp', 'flhBp',
            'fliAp1', 'fliDp', 'flgKp', 'fliCp', 'tarp', 'motAp' 'flgMp']

        self.flhDC_activated = [
            'fliLp1', 'fliEp1', 'fliFp1', 'flgAp', 'flgBp', 'flgEp', 'flhBp',
            'fliAp1', 'fliDp', 'flgKp']

        self.fliA_activated = [
            'fliCp', 'tarp', 'motAp', 'flgMp']

        flhDC_factors = {
            'fliLp1': {
                'flhDC': 1.2, 'fliA': 0.25},
            'fliEp1': {
                'flhDC': 0.45, 'fliA': 0.35},
            'fliFp1': {
                'flhDC': 0.35, 'fliA': 0.30},
            'flgBp': {
                'flhDC': 0.35, 'fliA': 0.45},
            'flgAp': {
                'flhDC': 0.15, 'fliA': 0.3},
            'flgEp': {
                'flhDC': 1.0, 'fliA': 4.0},
            'flhBp': {
                'flhDC': 0.1, 'fliA': 0.35},
            'fliAp1': {
                'flhDC': 1.0, 'fliA': 0.3},
            'fliDp': {
                'flhDC': 1.2, 'fliA': 0.25},
            'flgKp': {
                'flhDC': 1.2, 'fliA': 0.25}
        }

        def binary_sum_gates(promoter_factors):
            affinities = {}
            first, second = list(promoter_factors[
                list(promoter_factors.keys())[0]].keys())

            # this hard coding of simple addition is alarming and probably points
            # towards providing a function of promoter state for affinity rather
            # than a simple lookup of the affinity for each promoter state tuple.
            for promoter, factors in promoter_factors.items():
                affinities[(promoter, first, None)] = factors[first]
                affinities[(promoter, None, second)] = factors[second]
                affinities[(promoter, first, second)] = factors[first] + factors[second]

            return affinities

        # promoter affinities are binding affinity of RNAP onto promoter
        self.promoter_affinities = {
            ('flhDp', 'CRP'): 0.01}
        # self.promoter_affinities[('motAp', 'CpxR')] = 1.0
        flhDC_affinities = binary_sum_gates(flhDC_factors)
        self.promoter_affinities.update(flhDC_affinities)
        # for promoter in self.flhDC_activated:
        #     self.promoter_affinities[(promoter, 'flhDC')] = 1.0
        for promoter in self.fliA_activated:
            self.promoter_affinities[(promoter, 'fliA')] = 1.0
        self.promoter_affinities.update(
            parameters.get('promoter_affinities', {}))
        promoter_affinity_scaling = parameters.get('promoter_affinity_scaling', 1)
        self.promoter_affinities = {
            promoter: affinity * promoter_affinity_scaling
            for promoter, affinity in self.promoter_affinities.items()}

        self.transcripts = [
            (operon, product)
            for operon, products in self.chromosome_config['genes'].items()
            for product in products]

        self.protein_sequences = {
            (operon, product): self.knowledge_base.proteins[
                self.knowledge_base.genes[product]['id']]['seq']
            for operon, product in self.transcripts}

        self.transcript_templates = {
            key: generate_template(
                key,
                len(sequence),
                [key[1]])
            for key, sequence in self.protein_sequences.items()}

        # transcript affinities are the affinities transcripts to bind a ribosome and translate to protein
        # transcript affinities are scaled relative to the requirements to build a single full flagellum.
        self.min_tr_affinity = parameters.get('min_tr_affinity', 1e-1)
        tr_affinity_scaling = {
            'fliL': 2,
            'fliM': 34,
            'fliG': 26,
            'fliH': 12,
            'fliI': 6,
            'fliD': 5,
            'flgE': 120}
        self.transcript_affinities = {}
        for (operon, product) in self.transcripts:
            self.transcript_affinities[(operon, product)] = self.min_tr_affinity * tr_affinity_scaling.get(product,1)
        self.transcript_affinities.update(
            parameters.get('transcript_affinities', {}))


        self.transcription_factors = [
            'flhDC', 'fliA', 'CsgD', 'CRP', 'GadE', 'H-NS', 'CpxR', 'Fnr']

        self.complexation_monomer_ids = [
            'fliG', 'fliM', 'fliN', 'flhA', 'flhB', 'flhD', 'flhC', 'fliO',
            'fliP', 'fliQ', 'fliR', 'fliJ', 'fliI', 'fliH', 'fliL', 'flgH',
            'motA', 'motB', 'flgB', 'flgC', 'flgF', 'flgG', 'flgI', 'fliF',
            'fliE','fliC','flgL','flgK','fliD','flgE']

        self.complexation_complex_ids = [
            'flhDC',
            'flagellar motor switch',
            'flagella',
            'flagellar export apparatus subunit',
            'flagellar export apparatus',
            'flagellar hook',
            'flagellar motor']

        self.complexation_stoichiometry = {
            'flhDC': {
                'flhD': -4.0,
                'flhC': -2.0,
                'flhDC': 1.0
            },
            'flagellar motor switch reaction': {
                'flagellar motor switch': 1.0,
                'fliG': -26.0,
                'fliM': -34.0,
                'fliN': -1.0
            },
            'flagellar export apparatus reaction 1': {
                'flagellar export apparatus subunit': 1.0,
                'flhA': -1.0,
                'flhB': -1.0,
                'fliO': -1.0,
                'fliP': -1.0,
                'fliQ': -1.0,
                'fliR': -1.0,
                'fliJ': -1.0,
                'fliI': -6.0
            },
            'flagellar export apparatus reaction 2': {
                'flagellar export apparatus': 1.0,
                'flagellar export apparatus subunit': -1.0,
                'fliH': -12.0,
            },
            'flagellar motor reaction': {
                'flagellar motor': 1.0,
                'flagellar motor switch': -1.0,
                'fliL': -2.0,
                'flgH': -1.0,
                'motA': -1.0,
                'motB': -1.0,
                'flgB': -1.0,
                'flgC': -1.0,
                'flgF': -1.0,
                'flgG': -1.0,
                'flgI': -1.0,
                'fliF': -1.0,
                'fliE': -1.0,
            },
            'flagellar hook reaction': {
                'flagellar hook': 1,
                'flgE': -120.0,
            },
            'flagellum reaction': {
                'flagella': 1.0,
                'flagellar export apparatus': -1.0,
                'flagellar motor': -1.0,
                'fliC': -1.0,
                'flgL': -1.0,
                'flgK': -1.0,
                'fliD': -5.0,
                'flagellar hook': -1,
            }
        }

        reaction_default = 1e-4
        self.complexation_rates = {
            'flhDC': reaction_default,
            'flagellar motor switch reaction': reaction_default,
            'flagellar export apparatus reaction 1': reaction_default,
            'flagellar export apparatus reaction 2': reaction_default,
            'flagellar motor reaction': reaction_default,
            'flagellar hook reaction': reaction_default,
            'flagellum reaction': reaction_default}

