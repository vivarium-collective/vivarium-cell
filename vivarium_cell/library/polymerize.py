import random

import numpy as np
import logging as log

from vivarium.library.datum import Datum

INFINITY = float('inf')

def flatten(l):
    '''
    Flatten a list by one level:
        [[1, 2, 3], [[4, 5], 6], [7]] --> [1, 2, 3, [4, 5], 6, 7]
    '''

    return [
        item
        for sublist in l
        for item in sublist]

def add_merge(ds):
    '''
    Given a list of dicts, sum the values of each key.
    '''

    result = {}
    for d in ds:
        for key, value in d.items():
            if not key in result:
                result[key] = 0
            result[key] += value
    return result

def kinetics(E, S, kcat, km):
    return kcat * E * S / (S + km)

class Polymerase(Datum):
    defaults = {
        'id': 0,
        'state': None, # other states: ['bound', 'polymerizing', 'complete']
        'position': 0,
        'template': None,
        'template_index': 0,
        'terminator': 0}

    def __init__(self, config):
        super(Polymerase, self).__init__(config)

    def bind(self):
        self.state = 'bound'

    def start_polymerizing(self):
        self.state = 'occluding'

    def complete(self):
        self.state = 'complete'
        # print('completing polymerization: {}'.format(self.to_dict()))

    def is_bound(self):
        return self.state == 'bound'

    def is_polymerizing(self):
        return self.state == 'occluding' or self.state == 'polymerizing'

    def is_complete(self):
        return self.state == 'complete'

    def is_occluding(self):
        return self.state == 'bound' or self.state == 'occluding'

    def is_unoccluding(self, occlusion):
        return self.state == 'occluding' and self.position >= occlusion

    def unocclude(self):
        if self.state == 'occluding':
            self.state = 'polymerizing'

class BindingSite(Datum):
    defaults = {
        'position': 0,
        'length': 0,
        'thresholds': {}} # (factor, threshold)

    def __init__(self, config):
        super(BindingSite, self).__init__(config)

    def state_when(self, levels):
        '''
        Provide the binding state for the given levels of factors. 
        '''

        state = None
        for factor, threshold in self.thresholds.items():
            if levels[factor] >= threshold:
                state = factor
                break
        return state

class Terminator(Datum):
    defaults = {
        'position': 0,
        'strength': 0,
        'products': []}

    def __init__(self, config):
        super(Terminator, self).__init__(config)

    def between(self, before, after):
        return before < self.position < after or after < self.position < before

class Template(Datum):
    schema = {
        'sites': BindingSite,
        'terminators': Terminator}

    defaults = {
        'id': None,
        'position': 0,
        'direction': 1,
        'sites': [],
        'terminators': []}

    def __init__(self, config):
        super(Template, self).__init__(config)

        self.terminator_strength = 0
        for terminator in self.terminators:
            self.terminator_strength += terminator.strength

    def absolute_position(self, relative_position):
        return self.position + (relative_position * self.direction)

    def binding_state(self, levels):
        state = [
            site.state_when(levels)
            for site in self.sites]

        return tuple([self.id] + state)

    def strength_from(self, terminator_index):
        total = 0
        for index in range(terminator_index, len(self.terminators)):
            total += self.terminators[index].strength
        return total

    def next_terminator(self, position):
        for index, terminator in enumerate(self.terminators):
            if terminator.position * self.direction > position * self.direction:
                break
        return index

    def last_terminator(self):
        return self.terminators[-1]

    def terminates_at(self, index=0):
        if len(self.terminators[index:]) > 1:
            choice = random.random() * self.strength_from(index)
            return choice <= self.terminators[index].strength
        else:
            return True

    def choose_terminator(self, index=0):
        if len(self.terminators[index:]) > 1:
            choice = random.random() * self.strength_from(index)
            for terminator in self.terminators[index:]:
                if choice <= terminator.strength:
                    break
                else:
                    choice -= terminator.strength
            return terminator
        else:
            return self.terminators[index]

    def choose_product(self):
        terminator = self.choose_terminator()
        return terminator.products

    def products(self):
        return flatten([
            terminator.products
            for terminator in self.terminators])

def generate_template(id, length, products):
    return {
        'id': id,
        'position': 0,
        'direction': 1,
        'sites': [],
        'terminators': [
            {'position': length,
             'strength': 1.0,
             'products': products}]}

def all_products(templates):
    return list(set([
        product
        for template in templates.values()
        for product in template.products()]))

def template_products(config):
    return all_products({
        key: Template(config)
        for key, config in config.items()})

def polymerize_step(
        sequences,
        polymerases,
        templates,
        symbol_to_monomer,
        monomer_limits):

    complete_polymers = {
        product: 0
        for product in all_products(templates)}

    monomers = {
        monomer: 0
        for monomer in monomer_limits.keys()}
    terminated = 0

    for polymerase in polymerases.values():
        if polymerase.is_polymerizing():
            template = templates[polymerase.template]
            projection = polymerase.position + 1

            try:
                monomer_symbol = sequences[template.id][polymerase.position]
            except IndexError as e:
                log.error('index beyond sequence: polymerase - {} template - {}'.format(
                    polymerase,
                    template))
                monomer_symbol = random.choice(list(symbol_to_monomer.keys()))

            monomer = symbol_to_monomer[monomer_symbol]

            if monomer_limits[monomer] > 0:
                monomer_limits[monomer] -= 1
                monomers[monomer] += 1
                polymerase.position = projection
                absolute_position = template.absolute_position(
                    polymerase.position)

                terminator = template.terminators[polymerase.terminator]
                if terminator.position == absolute_position:
                    if template.terminates_at(polymerase.terminator):
                        polymerase.complete()
                        terminated += 1

                        for product in terminator.products:
                            complete_polymers[product] += 1
                    else:
                        polymerase.terminator += 1

    polymerases = {
        id: polymerase
        for id, polymerase in polymerases.items()
        if not polymerase.is_complete()}

    return monomers, monomer_limits, terminated, complete_polymers, polymerases
    

def polymerize_to(
        sequences,
        polymerases,
        templates,
        additions,
        symbol_to_monomer,
        monomer_limits):

    for step in range(additions):
        monomers, monomer_limits, terminated, complete_polymers, polymerases = polymerize_step(
            sequences, polymerases, templates, symbol_to_monomer, monomer_limits)
    return monomers, monomer_limits, terminated, complete_polymers, polymerases


class Elongation(object):
    def __init__(
            self,
            sequence,
            templates,
            limits,
            symbol_to_monomer,
            elongation=0):
        self.sequence = sequence
        self.templates = templates
        self.time = 0
        self.monomers = {}
        self.symbol_to_monomer = symbol_to_monomer
        self.complete_polymers = {}
        self.previous_elongations = int(elongation)
        self.elongation = elongation
        self.limits = limits

    def step(self, interval, limits, polymerases):
        self.time += interval
        monomers, limits, terminated, complete, polymerases = polymerize_step(
            self.sequence,
            polymerases,
            self.templates,
            self.symbol_to_monomer,
            limits)

        self.monomers = add_merge([self.monomers, monomers])
        self.complete_polymers = add_merge([
            self.complete_polymers, complete])

        return terminated, limits, polymerases

    def store_partial(self, interval):
        self.elongation += interval

    def elongate_to(self, now, rate, limits, polymerases):
        '''
        Track increments of time and accumulate partial elongations, emitting the full
        elongation once a unit is attained.

        Returns number of polymerases that terminated this step, and the updated 
        monomer limits after all elongations.
        '''

        progress = rate * (now - self.time)
        self.elongation += progress
        elongations = int(self.elongation) - self.previous_elongations
        self.time = now
        terminated = 0

        if elongations:
            monomers, limits, terminated, complete, polymerases = polymerize_to(
                self.sequence,
                polymerases,
                self.templates,
                elongations,
                self.symbol_to_monomer,
                limits)
            self.monomers = add_merge([self.monomers, monomers])
            self.complete_polymers = add_merge([
                self.complete_polymers, complete])
            self.previous_elongations = int(self.elongation)

        return terminated, limits, polymerases

    def complete(self):
        return len(self.complete_polymers)


def build_double_stoichiometry(promoter_count):
    '''
    Builds a stoichiometry for the given promoters. There are two states per promoter,
    open and bound, and two reactions per promoter, binding and unbinding. In addition
    there is a single substrate for available RNAP in the final index.

    Here we are assuming
    '''
    stoichiometry = np.zeros((promoter_count * 2, promoter_count * 2 + 1), dtype=np.int64)
    for index in range(promoter_count):
        # forward reaction
        stoichiometry[index][index] = -1
        stoichiometry[index][index + promoter_count] = 1
        stoichiometry[index][-1] = -1 # forward reaction consumes RNAP also

        # reverse reaction
        stoichiometry[index + promoter_count][index] = 1
        stoichiometry[index + promoter_count][index + promoter_count] = -1

    return stoichiometry

def build_double_rates(affinities, advancement):
    return np.concatenate([
        affinities,
        np.repeat(advancement, len(affinities))])

def build_stoichiometry(promoter_count):
    '''
    Builds a stoichiometry for the given promoters. There are two states per promoter,
    open and bound, and two reactions per promoter, binding and unbinding. In addition
    there is a single substrate for available RNAP in the final index.

    Here we are assuming
    '''
    stoichiometry = np.zeros((promoter_count, promoter_count * 2 + 1), dtype=np.int64)
    for index in range(promoter_count):
        # forward reaction
        stoichiometry[index][index] = -1
        stoichiometry[index][index + promoter_count] = 1
        stoichiometry[index][-1] = -1 # forward reaction consumes RNAP also

    return stoichiometry

