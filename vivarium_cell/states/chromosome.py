import random
import copy
import numpy as np

from vivarium.library.datum import Datum
from vivarium_cell.data.chromosomes.toy_chromosome import toy_chromosome_config
from vivarium_cell.data.nucleotides import nucleotides
from vivarium_cell.library.polymerize import Polymerase, BindingSite, Terminator, Template, Elongation, polymerize_to, add_merge

INFINITY = float('inf')

def frequencies(l):
    '''
    Return number of times each item appears in the list.
    '''

    result = {}
    for item in l:
        if not item in result:
            result[item] = 0
        result[item] += 1
    return result

def rna_bases(sequence):
    sequence = sequence.replace('A', 'U')
    sequence = sequence.replace('T', 'A')
    sequence = sequence.replace('C', 'x')
    sequence = sequence.replace('G', 'C')
    sequence = sequence.replace('x', 'G')
    return sequence

def sequence_monomers(sequence, begin, end):
    if begin < end:
        subsequence = sequence[begin:end]
    else:
        subsequence = sequence[begin:end:-1]
    return subsequence

def traverse(tree, key, f, combine):
    '''
    Traverse the given tree starting using the `key` node as the root and calling `f` on each leaf,
    combining values with `combine` at each subsequent level to create new leaves for `f`.
    '''

    node = tree[key]
    if node.children:
        eldest = traverse(tree, node.children[0], f, combine)
        youngest = traverse(tree, node.children[1], f, combine)
        outcome = combine(eldest, youngest)
        return f(node, outcome)
    else:
        return f(node)


class Operon(Datum):
    defaults = {
        'id': '',
        'position': 0,
        'direction': 1,
        'length': 0,
        'genes': []}

    def __init__(self, config):
        super(Operon, self).__init__(config)

class Domain(Datum):
    defaults = {
        'id': 0,
        'lead': 0,
        'lag': 0,
        'children': []}

    def __init__(self, config):
        super(Domain, self).__init__(config)

    def contains(self, position):
        if position < 0:
            return position < self.lag
        else:
            return position > self.lead

    def surpassed(self, position, lead, lag):
        if position < 0:
            return position <= self.lag and position > self.lag + lag
        else:
            return position >= self.lead and position < self.lead + lead

    def random_child(self):
        return random.choice(self.children)

    def descendants(self, tree):
        return [self] + [tree[child].descendants(tree) for child in self.children]

class RnapTerminator(Terminator):
    def operon_from(self, genes, promoter):
        return Operon({
            'id': self.products[0], # assume there is only one operon product
            'position': promoter.position,
            'direction': promoter.direction,
            'length': (self.position - promoter.position) * promoter.direction,
            'genes': genes.get(self.products[0], [])})

class Promoter(Template):
    '''
    Promoters are the main unit of expression. They define a direction of
    polymerization, contain binding sites for transcription factors, and declare some
    number of terminators, each of which has a strength and corresponds to a particular
    operon if chosen.
    '''

    schema = {
        'sites': BindingSite,
        'terminators': RnapTerminator}

    def operons(self, genes):
        return [
            terminator.operon_from(genes, self)
            for terminator in self.terminators]

class Rnap(Polymerase):
    defaults = {
        'id': 0,
        'template': None,
        'template_index': 0,
        'terminator': 0,
        'domain': 0,
        'state': None, # other states: ['bound', 'polymerizing', 'occluding', 'complete']
        'position': 0}

    def __init__(self, config):
        super(Rnap, self).__init__(config)

class Chromosome(Datum):
    schema = {
        'promoters': Promoter,
        'domains': Domain,
        'rnaps': Rnap}

    defaults = {
        'sequence': '',
        'genes': {},
        'promoters': {},
        'promoter_order': [],
        'domains': {
            0: {
                'id': 0,
                'lead': 0,
                'lag': 0,
                'children': []}},
        'root_domain': 0,
        'rnap_id': 0,
        'rnaps': {}}

    def operons(self):
        return [
            operon
            for promoter in self.promoters.values()
            for operon in promoter.operons(self.genes)]

    def copy_number(self, position, domain_key=None):
        if not domain_key:
            domain_key = self.root_domain
        domain = self.domains[domain_key]
        if domain.contains(position):
            return 1
        else:
            return sum([
                self.copy_number(position, child)
                for child in domain.children])

    def promoter_copy_numbers(self):
        copy_numbers = [
            self.copy_number(self.promoters[promoter_key].position)
            for promoter_key in self.promoter_order]
        return np.array(copy_numbers)

    def promoter_rnaps(self):
        by_promoter = {
            promoter_key: {}
            for promoter_key in self.promoter_order}

        for rnap in self.rnaps.values():
            if rnap.is_occluding():
                by_promoter[rnap.template][rnap.domain] = rnap

        return by_promoter

    def promoter_domains(self):
        return {
            promoter_key: self.position_domains(
                self.root_domain,
                self.promoters[promoter_key].position)
            for promoter_key in self.promoter_order}

    def position_domains(self, domain_index, position):
        domain = self.domains[domain_index]
        if len(domain.children) == 0 or (position < 0 and domain.lag >= position) or (position >= 0 and domain.lead <= position):
            return set([domain_index])
        else:
            return set.union(*[
                self.position_domains(child, position)
                for child in domain.children])

    def apply_thresholds(self, thresholds):
        for path, level in thresholds.items():
            promoter, factor = path
            found = False
            for site in self.promoters[promoter].sites:
                if factor in site.thresholds:
                    site.thresholds[factor] = level
                    found = True
            if not found:
                print('binding site not found for {} with level {}'.format(path, level))

    def bind_rnap(self, promoter_index, domain):
        self.rnap_id += 1
        promoter_key = self.promoter_order[promoter_index]

        new_rnap = Rnap({
            'id': self.rnap_id,
            'template': promoter_key,
            'template_index': promoter_index,
            'domain': domain,
            'position': 0})
        new_rnap.bind()
        self.rnaps[new_rnap.id] = new_rnap
        return new_rnap

    def terminator_distance(self):
        distance = INFINITY
        for rnap in self.rnaps.values():
            if rnap.is_polymerizing():
                promoter = self.promoters[rnap.template]

                # rnap position is relative to the promoter it is bound to
                rnap_position = promoter.absolute_position(rnap.position)
                terminator_index = promoter.next_terminator(rnap_position)
                rnap.terminator = terminator_index
                terminator = promoter.terminators[terminator_index]
                span = abs(terminator.position - rnap_position)
                if span < distance:
                    distance = span

        if distance == INFINITY:
            distance = 1
        return distance

    def sequences(self):
        return {
            promoter_key: rna_bases(sequence_monomers(
                self.sequence,
                promoter.position,
                promoter.last_terminator().position))
            for promoter_key, promoter in self.promoters.items()}

    def product_sequences(self):
        sequences = {}
        for promoter_key, promoter in self.promoters.items():
            for terminator in promoter.terminators:
                for product in terminator.products:
                    sequences[product] = rna_bases(sequence_monomers(
                        self.sequence,
                        promoter.position,
                        terminator.position))

        return sequences

    def next_polymerize(self, elongation_limit=INFINITY, monomer_limits={}):
        distance = self.terminator_distance()
        elongate_to = min(elongation_limit, distance)

        sequences = self.sequences()

        monomers, monomer_limits, terminated, complete_transcripts, self.rnaps = polymerize_to(
            sequences,
            self.rnaps,
            self.promoters,
            elongate_to,
            nucleotides,
            monomer_limits)

        return elongate_to, monomers, monomer_limits, complete_transcripts

    def polymerize(self, elongation, monomer_limits):
        iterations = 0
        attained = 0
        all_monomers = {}
        complete_transcripts = {}

        while attained < elongation:
            elongated, monomers, monomer_limits, complete = self.next_polymerize(
                elongation_limit=elongation - attained,
                monomer_limits=monomer_limits)
            attained += elongated
            all_monomers = add_merge([all_monomers, monomers])
            complete_transcripts = add_merge([complete_transcripts, complete])
            iterations += 1

        return iterations, all_monomers, complete_transcripts, monomer_limits

    def initiate_replication(self):
        leaves = [leaf for leaf in self.domains.values() if not leaf.children]
        next_id = max([leaf.id for leaf in leaves]) + 1
        for leaf in leaves:
            for child in [0, 1]:
                domain = Domain({'id': next_id + child})
                self.domains[domain.id] = domain
            leaf.children = [next_id, next_id + 1]
            next_id += 2

    def advance_replisomes(self, distances):
        '''
        distances is a dictionary of domain ids to tuples of how far each strand advances
        of the form (lead, lag)
        '''
        for domain_key, distance in distances.items():
            domain = self.domains[domain_key]
            lead, lag = distances[domain_key]

            for rnap in self.rnaps.values():
                if rnap.domain == domain_key:
                    promoter = self.promoters[rnap.template]
                    position = promoter.position
                    position += rnap.position * promoter.direction
                    if domain.surpassed(position, lead, -lag):
                        rnap.domain = domain.random_child()

            domain.lead += lead
            domain.lag -= lag

    def divide_chromosome(self, domain, division=None):
        if not division:
            division = {
                'sequence': self.sequence,
                'promoters': {id: promoter.to_dict() for id, promoter in self.promoters.items()},
                'domains': {domain.id: domain.to_dict()},
                'root_domain': domain.id,
                'rnaps': {
                    rnap.id: rnap.to_dict()
                    for rnap in self.rnaps.items()
                    if rnap.domain == domain.id}}

        else:
            division['domains'][domain.id] = domain.to_dict()
            for rnap in self.rnaps.values():
                if rnap.domain == domain.id:
                    division['rnaps'].append(rnap.to_dict())

        return division

    def combine_state(self, a, b):
        merge = copy.deepcopy(a)
        merge['domains'].update(b['domains'])
        merge['rnaps'].extend(b['rnaps'])
        return merge

    def terminate_replication(self):
        children = self.domains[self.root_domain].children
        divided = [
            traverse(
                self.domains,
                child,
                self.divide_chromosome,
                self.combine_state)
            for child in children]

        return [Chromosome(fork) for fork in divided]

    def __init__(self, config):
        super(Chromosome, self).__init__(config)
        if self.promoter_order:
            self.promoter_order = list(self.promoters.keys())



def test_chromosome():
    chromosome = Chromosome(toy_chromosome_config)
    print(chromosome.promoters['pA'].terminators[0].products)
    print(chromosome)

    print('operons:')
    print(chromosome.operons())

    chromosome.initiate_replication()
    print(chromosome.domains)
    assert len(chromosome.domains) == 3

    chromosome.advance_replisomes({0: (5, 7)})
    print('replisomes:')
    print(chromosome)

    # chromosome.initiate_replication()
    # print(chromosome.domains)
    # assert len(chromosome.domains) == 7

    # chromosome.advance_replisomes({0: (7, 5), 1: (3, 4), 2: (8, 9)})
    # print('replisomes:')
    # print(chromosome)

    print('copy numbers 1 4 7 -9 11')
    print(chromosome.copy_number(1))
    print(chromosome.copy_number(4))
    print(chromosome.copy_number(7))
    print(chromosome.copy_number(-9))
    print(chromosome.copy_number(11))

    print('promoter copy numbers')
    print(chromosome.promoter_copy_numbers())

    print('promoter rnaps')
    print(chromosome.promoter_rnaps())

    print('promoter domains')
    print(chromosome.promoter_domains())

    print('rnaps')
    print([rnap.to_dict() for rnap in chromosome.rnaps.values()])

    print('completed after advancing 5')
    print(chromosome.polymerize(5, {
        'rATP': 100, 'rUTP': 100, 'rGTP': 100, 'rCTP': 100}))

    print('rnaps after polymerizing')
    print(chromosome.rnaps)
    
    children = chromosome.terminate_replication()
    print('termination:')
    print(children)

    return chromosome

if __name__ == '__main__':
    test_chromosome()
