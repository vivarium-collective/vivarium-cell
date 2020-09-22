from __future__ import absolute_import, division, print_function

import cobra
import cobra.test
from cobra.medium import minimal_medium

from cobra import Model, Reaction, Metabolite, Configuration
from vivarium.library.units import units
from vivarium_cell.processes.derive_globals import AVOGADRO
from vivarium_cell.data.synonyms import get_synonym


EXTERNAL_PREFIX = 'EX_'
DEFAULT_UPPER_BOUND = 100.0

def build_model(stoichiometry, reversible, objective, external_molecules, default_upper_bound=1000):
    model = Model('fba')
    model.compartments = {'c': 'cytoplasm'}

    metabolite_keys = {}
    for reaction_key, chemistry in stoichiometry.items():
        metabolite_keys.update(chemistry)

    metabolites = {
        metabolite: Metabolite(metabolite, name=metabolite, compartment='c')
        for metabolite in list(metabolite_keys.keys())}

    model.add_metabolites(metabolites.values())

    # make reactions
    reactions = {}
    for reaction_key, chemistry in stoichiometry.items():
        reaction = Reaction(reaction_key, name=reaction_key)

        # set reaction bounds
        reaction.upper_bound = default_upper_bound
        if reaction_key in reversible:
            reaction.lower_bound = -reaction.upper_bound

        # make stoichiometry
        reaction_model = {
            metabolites[metabolite]: value
            for metabolite, value in chemistry.items()}
        reaction.add_metabolites(reaction_model)

        reactions[reaction_key] = reaction

    # make exchange reactions for all external_molecules
    for external in external_molecules:
        external_key = EXTERNAL_PREFIX + external
        reaction = Reaction(external_key, name=external_key)

        # set reaction bounds
        reaction.upper_bound = default_upper_bound
        reaction.lower_bound = -default_upper_bound  # TODO -- should exchanges have symmetric bounds by default?

        # make stoichiometry
        reaction_model = {metabolites[external]: -1}
        reaction.add_metabolites(reaction_model)

        reactions[external_key] = reaction

    model.add_reactions(reactions.values())

    model.objective = {
        reactions[reaction_key]: value
        for reaction_key, value in objective.items()}

    return model

def extract_model(model):
    """
    TODO -- where do demands and sinks go?
    demands = model.demands
    sinks = model.sinks

    # boundary reactions include exchanges, demands, sinks
    boundary = model.boundary
    boundary_reactions = [reaction.id for reaction in boundary]
    """

    reactions = model.reactions
    metabolites = model.metabolites
    exchanges = model.exchanges
    objective_expression = model.objective.expression.args

    # get stoichiometry and flux bounds
    stoichiometry = {}
    flux_bounds = {}
    reversible = []
    for reaction in reactions:
        reaction_id = reaction.id
        reaction_metabolites = reaction.metabolites
        bounds = list(reaction.bounds)
        stoichiometry[reaction.id] = {
            metabolite.id: coeff for metabolite, coeff in reaction_metabolites.items()}
        flux_bounds[reaction_id] = bounds
        if not any(b == 0.0 for b in bounds):
            reversible.append(reaction_id)

    # get external molecules and exchange bounds from exchanges
    external_molecules = []
    exchange_bounds = {}
    for reaction in exchanges:
        reaction_metabolites = list(reaction.metabolites.keys())
        assert len(reaction_metabolites) == 1  # only 1 molecule in the exchange reaction
        metabolite_id = reaction_metabolites[0].id
        external_molecules.append(metabolite_id)
        bounds = list(reaction.bounds)
        exchange_bounds[metabolite_id] = bounds

    # get molecular weights
    molecular_weights = {}
    nonexisting_elements = ['R', 'X']
    for metabolite in metabolites:
        if any([e in nonexisting_elements for e in metabolite.elements]):
            continue
        molecular_weights[metabolite.id] = metabolite.formula_weight

    # get objective
    objective = {}
    for expression in objective_expression:
        exp_str = str(expression)
        coeff, reaction_id = exp_str.split('*')
        try:
            reactions.get_by_id(reaction_id)
            objective[reaction_id] = float(coeff)
        except:
            pass

    # get flux scaling factor based on the objective's predicted added mass
    # this adjusts the BiGG FBA bounds to approximate single-cell rates
    target_added_mass = 4.9e-7  # fit to approximate a doubling time of 2520 sec (42 min) in iAF1260b

    solution = model.optimize()
    objective_value = solution.objective_value
    added_mass = 0
    for reaction_id, coeff1 in objective.items():
        for mol_id, coeff2 in stoichiometry[reaction_id].items():
            if coeff2 < 0: # molecule is used to make biomass (negative coefficient)
                mw = molecular_weights.get(mol_id) * (units.g / units.mol)
                count = int(-coeff1 * coeff2 * objective_value)
                mol = count / AVOGADRO
                mol_added_mass = mw * mol
                added_mass += mol_added_mass.to('fg').magnitude

    flux_scaling = target_added_mass / added_mass

    return {
        'stoichiometry': stoichiometry,
        'reversible': reversible,
        'external_molecules': external_molecules,
        'objective': objective,
        'flux_bounds': flux_bounds,
        'exchange_bounds': exchange_bounds,
        'molecular_weights': molecular_weights,
        'flux_scaling': flux_scaling}

def swap_synonyms(model):
    metabolites = model.metabolites

    # swap metabolite ids
    for mol in metabolites:
        mol_id = mol.id
        mol_id = get_synonym(mol_id)
        mol.id = mol_id

class CobraFBA(object):
    """
    This class provides an interface to cobra FBA.
    It can load in BiGG models (http://bigg.ucsd.edu/models) if provided a model_path to a saved JSON BiGG model,
    or load in a novel model specified by stoichiometry, reversibility, and objective.

    TODO (Eran) -- MOMA option is provided, but has not yet been tested.
    """

    cobra_configuration = Configuration()

    def __init__(self, config={}):
        model_path = config.get('model_path')

        # get tolerances
        self.default_tolerance = config.get('default_tolerance', [0.95, 1])
        self.tolerance = config.get('tolerance', {})

        # set MOMA (minimization of metabolic adjustment)
        self.moma = config.get('moma', False)

        if model_path:
            # load a BiGG model
            self.model = cobra.io.load_json_model(model_path)
            swap_synonyms(self.model)
            extract = extract_model(self.model)

            self.stoichiometry = extract['stoichiometry']
            self.reversible = extract['reversible']
            self.external_molecules = extract['external_molecules']
            self.objective = extract['objective']
            self.flux_bounds = extract['flux_bounds']
            self.molecular_weights = extract['molecular_weights']
            self.exchange_bounds = extract['exchange_bounds']
            self.default_upper_bound = DEFAULT_UPPER_BOUND  # TODO -- can this be extracted from model?
            self.flux_scaling = extract['flux_scaling']

        else:
            # create an FBA model from config
            self.stoichiometry = config['stoichiometry']
            self.reversible = config.get('reversible', [])
            self.external_molecules = config['external_molecules']
            self.objective = config['objective']
            self.flux_bounds = config.get('flux_bounds', {})
            self.molecular_weights = config.get('molecular_weights', {})
            self.exchange_bounds = config.get('exchange_bounds', {})
            self.default_upper_bound = config.get('default_upper_bound', DEFAULT_UPPER_BOUND)
            self.flux_scaling = config.get('flux_scaling', 1)

            self.model = build_model(
                self.stoichiometry,
                self.reversible,
                self.objective,
                self.external_molecules,
                self.default_upper_bound)

            # apply constraints
            self.constrain_reaction_bounds(self.flux_bounds)
            self.set_exchange_bounds()

        self.exchange_bounds_keys = list(self.exchange_bounds.keys())

        # get minimal external state
        # TODO -- make sure that scaling is accounted for
        max_growth = self.model.slim_optimize()
        max_exchange = minimal_medium(self.model, max_growth)
        self.minimal_external = {ex[len(EXTERNAL_PREFIX):len(ex)]: value
            for ex, value in max_exchange.items()}

        # initialize solution
        self.solution = self.model.optimize()

    def set_exchange_bounds(self, bounds={}):
        '''
        apply new_bounds for the defined molecules.
        reset unincluded molecules to their exchange_bounds.
        '''
        for external_mol, level in self.exchange_bounds.items():
            reaction = self.model.reactions.get_by_id(EXTERNAL_PREFIX + external_mol)

            if external_mol in bounds:
                level = bounds[external_mol] / self.flux_scaling

            if type(level) is list:
                reaction.upper_bound = level[1]
                reaction.lower_bound = level[0]
            elif isinstance(level, int) or isinstance(level, float):
                # reaction.upper_bound = level
                reaction.lower_bound = level

    def constrain_flux(self, bounds={}):
        '''add externally imposed constraints'''
        for reaction_id, bound in bounds.items():
            reaction = self.model.reactions.get_by_id(reaction_id)
            scaled_level = bound / self.flux_scaling

            if EXTERNAL_PREFIX in reaction_id and \
                any(substring in reaction_id for substring in self.exchange_bounds_keys):
                # exchanges use reverse flux
                scaled_level *= -1

            if reaction_id in self.tolerance:
                # use configured tolerance
                lower_tolerance, upper_tolerance = self.tolerance[reaction_id]
                reaction.upper_bound = upper_tolerance * scaled_level
                reaction.lower_bound = lower_tolerance * scaled_level
            else:
                # use default tolerance
                if bound >= 0:
                    reaction.upper_bound = self.default_tolerance[1] * scaled_level
                    reaction.lower_bound = self.default_tolerance[0] * scaled_level
                else:
                    reaction.upper_bound = self.default_tolerance[0] * scaled_level
                    reaction.lower_bound = self.default_tolerance[1] * scaled_level

    def constrain_reaction_bounds(self, bounds={}):
        reactions = self.get_reactions(list(bounds.keys()))
        for reaction_id, bound in bounds.items():
            reaction = reactions[reaction_id]
            scaled_bound = [b / self.flux_scaling for b in bound]
            reaction.lower_bound, reaction.upper_bound = scaled_bound

    def regulate_flux(self, reactions):
        '''regulate flux based on True/False activity values for each id in reactions dictionary'''
        for reaction_id, activity in reactions.items():
            reaction = self.model.reactions.get_by_id(reaction_id)

            if not activity:
                # no activity. reaction flux set to 0
                reaction.upper_bound = 0.0
                reaction.lower_bound = 0.0
            elif activity and reaction.bounds == (0.0, 0.0):
                # if new bounds need to be set
                if reaction_id in self.flux_bounds:
                    bounds = self.flux_bounds[reaction_id]
                    scaled_bounds = [b / self.flux_scaling for b in bounds]
                    reaction.lower_bound, reaction.upper_bound = scaled_bounds
                elif reaction_id in self.reversible:
                    reaction.upper_bound = self.default_upper_bound / self.flux_scaling
                    reaction.lower_bound = -self.default_upper_bound / self.flux_scaling
                else:
                    # set bounds based on default
                    reaction.upper_bound = self.default_upper_bound / self.flux_scaling
                    reaction.lower_bound = 0.0

    def objective_value(self):
        if self.solution:
            objective_value = self.solution.objective_value * self.flux_scaling
            return objective_value
        else:
            return float('nan')

    def optimize(self):

        if self.moma:
            self.solution = cobra.flux_analysis.moma(self.model, solution=self.solution)
        else:
            self.solution = self.model.optimize()

        return self.objective_value()

    def external_reactions(self):
        return [
            EXTERNAL_PREFIX + molecule
            for molecule in self.external_molecules]

    def internal_reactions(self):
        all_reactions = set(self.reaction_ids())
        return all_reactions - set(self.external_reactions())

    def read_fluxes(self, molecules):
        return {
            molecule: self.solution.fluxes[molecule] * self.flux_scaling
            for molecule in molecules}

    def read_internal_fluxes(self):
        return self.read_fluxes(self.internal_reactions())

    def read_exchange_reactions(self):
        '''leaves the prefix in exchange reactions keys'''
        return self.read_fluxes(self.external_reactions())

    def read_exchange_fluxes(self):
        '''removes the prefix from exchange reactions keys, leaving only the external molecule id'''
        external = self.external_reactions()
        levels = self.read_fluxes(external)
        return {
            molecule[len(EXTERNAL_PREFIX):len(molecule)]: level
            for molecule, level in levels.items()}

    def reaction_ids(self):
        return [reaction.id for reaction in self.model.reactions]

    def get_reactions(self, reactions=[]):
        if not reactions:
            reactions = self.reaction_ids()

        return {
            reaction: self.model.reactions.get_by_id(reaction)
            for reaction in reactions}

    def get_reaction_bounds(self, reactions=[]):
        return {
            reaction_key: (reaction.lower_bound, reaction.upper_bound)
            for reaction_key, reaction in self.get_reactions(reactions).items()}



def test_minimal():
    stoichiometry = {
        'R1': {'A': -1, 'B': 1},
        'EB': {'B': -1}}

    objective = {'EB': 1.0}

    external_molecules = ['A']

    initial_state = {
        'A': 5}

    fba = CobraFBA({
        'stoichiometry': stoichiometry,
        'reversible': list(stoichiometry.keys()),
        'objective': objective,
        'external_molecules': external_molecules,
        'initial_state': initial_state,
        })

    fba.set_exchange_bounds(initial_state)

    return fba

def test_fba():
    stoichiometry = {
        'R1': {'A': -1, 'ATP': -1, 'B': 1},
        'R2a': {'B': -1, 'ATP': 2, 'NADH': 2, 'C': 1},
        'R2b': {'C': -1, 'ATP': -2, 'NADH': -2, 'B': 1},
        'R3': {'B': -1, 'F': 1},
        'R4': {'C': -1, 'G': 1},
        'R5': {'G': -1, 'C': 0.8, 'NADH': 2},
        'R6': {'C': -1, 'ATP': 2, 'D': 3},
        'R7': {'C': -1, 'NADH': -4, 'E': 3},
        'R8a': {'G': -1, 'ATP': -1, 'NADH': -2, 'H': 1},
        'R8b': {'G': 1, 'ATP': 1, 'NADH': 2, 'H': -1},
        'Rres': {'NADH': -1, 'O2': -1, 'ATP': 1},
        'v_biomass': {'C': -1, 'F': -1, 'H': -1, 'ATP': -10, 'BIOMASS': 1}
    }

    external_molecules = ['A', 'F', 'D', 'E', 'H', 'O2', 'BIOMASS']

    objective = {'v_biomass': 1.0}

    initial_state = {
        'internal': {
            'mass': 1339,
            'volume': 1E-15},
        'external': {
            'A': 21.0,
            'F': 5.0,
            'D': 12.0,
            'E': 12.0,
            'H': 5.0,
            'O2': 100.0}}

    fba = CobraFBA({
        'stoichiometry': stoichiometry,
        'reversible': list(stoichiometry.keys()),
        'objective': objective,
        'external_molecules': external_molecules})

    fba.set_exchange_bounds(initial_state['external'])

    return fba

class JsonFBA(object):
    def __init__(self, path):
        self.model = cobra.io.load_json_model(path)

def test_canonical():
    fba = JsonFBA('vivarium_cell/bigg_models/e_coli_core.json')
    return fba

def test_demo():
    model = Model('example_model')

    reaction = Reaction('3OAS140')
    reaction.name = '3 oxoacyl acyl carrier protein synthase n C140 '
    reaction.subsystem = 'Cell Envelope Biosynthesis'
    reaction.lower_bound = 0.  # This is the default
    reaction.upper_bound = 1000.  # This is the default

    ACP_c = Metabolite(
        'ACP_c',
        formula='C11H21N2O7PRS',
        name='acyl-carrier-protein',
        compartment='c')
    omrsACP_c = Metabolite(
        '3omrsACP_c',
        formula='C25H45N2O9PRS',
        name='3-Oxotetradecanoyl-acyl-carrier-protein',
        compartment='c')
    co2_c = Metabolite('co2_c', formula='CO2', name='CO2', compartment='c')
    malACP_c = Metabolite(
        'malACP_c',
        formula='C14H22N2O10PRS',
        name='Malonyl-acyl-carrier-protein',
        compartment='c')
    h_c = Metabolite('h_c', formula='H', name='H', compartment='c')
    ddcaACP_c = Metabolite(
        'ddcaACP_c',
        formula='C23H43N2O8PRS',
        name='Dodecanoyl-ACP-n-C120ACP',
        compartment='c')

    reaction.add_metabolites({
        malACP_c: -1.0,
        h_c: -1.0,
        ddcaACP_c: -1.0,
        co2_c: 1.0,
        ACP_c: 1.0,
        omrsACP_c: 1.0
    })

    print(reaction.reaction)  # This gives a string representation of the reaction

    reaction.gene_reaction_rule = '( STM2378 or STM1197 )'
    print(reaction.genes)

    model.add_reactions([reaction])

    # Now there are things in the model
    print('%i reaction' % len(model.reactions))
    print('%i metabolites' % len(model.metabolites))
    print('%i genes' % len(model.genes))

    # Iterate through the the objects in the model
    print('Reactions')
    print('---------')
    for x in model.reactions:
        print('%s : %s' % (x.id, x.reaction))

    print('')
    print('Metabolites')
    print('-----------')
    for x in model.metabolites:
        print('%9s : %s' % (x.id, x.formula))

    print('')
    print('Genes')
    print('-----')
    for x in model.genes:
        associated_ids = (i.id for i in x.reactions)
        print('%s is associated with reactions: %s' %
              (x.id, '{' + ', '.join(associated_ids) + '}'))

    model.objective = '3OAS140'

    print(model.objective.expression)
    print(model.objective.direction)

    class DemoFBA(object):
        def __init__(self, model):
            self.model = model

    return DemoFBA(model)


if __name__ == '__main__':
    fba = test_fba()
    # fba = test_minimal()
    # fba = test_canonical()
    # fba = test_demo()
    # fba = test_test()

    # cobra.io.save_json_model(fba.model, 'demo_model.json')
    print('MODEL: {}'.format(fba.model))
    print('REACTIONS: {}'.format(fba.model.reactions))
    print('METABOLITES: {}'.format(fba.model.metabolites))
    print('GENES: {}'.format(fba.model.genes))
    print('COMPARTMENTS: {}'.format(fba.model.compartments))
    print('SOLVER: {}'.format(fba.model.solver))
    print('EXPRESSION: {}'.format(fba.model.objective.expression))

    print(fba.optimize())
    print(fba.model.summary())
    print('internal: {}'.format(fba.internal_reactions()))
    print('external: {}'.format(fba.external_reactions()))
    print(fba.reaction_ids())
    print(fba.get_reactions())
    print(fba.get_reaction_bounds())
    print(fba.read_exchange_fluxes())
