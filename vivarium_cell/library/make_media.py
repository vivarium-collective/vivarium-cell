'''
Functions for making media
'''

from __future__ import absolute_import, division, print_function

import uuid

from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
from vivarium.library.units import units

# Raw data class
from vivarium_cell.data.knowledge_base import KnowledgeBase

INF = float("inf")
NEG_INF = float("-inf")

COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS

def cmp(a, b):
	return (a > b) - (a < b)

grammar = Grammar(r"""
    timeline = one_event*
    one_event = numeric? ws? recipe break?
    recipe = one_ingredient*
    one_ingredient = add? subtract? ingredient
    ingredient = media_id counts? volume? inf?
    counts = ws numeric count_key
    volume = ws numeric vol_key
    count_key = ws "mmol" ws?
    vol_key = ws "L" ws?
    inf = ws "Infinity" ws?
    media_id = ~"[A-Za-z0-9_][^,\s]+"i
    numeric = ~"[0-9.]+"i
    add = ws? "+" ws?
    subtract = ws? "-" ws?
    break = ws? "," ws?
    ws  = ~"\s*"
    """)

class AddIngredientsError(Exception):
    pass

class Media(object):
    '''
    A media object is a factory for making new media by combining
    media and ingredients at different volumes (with self.make_recipe().
    Ingredients can be added by either specifying their weight (in grams)
    or the counts (in mmol) in addition to the volume. The new media dicts
    are returned to the caller, and are not saved in this object. A media
    object holds dicts about stock media in ```self.stock_media``` and the
    formula weight of environmental molecules in
    ```self.environment_molecules_fw```, which is needed for mixing in
    ingredients at weights.
    '''

    def __init__(self):

        raw_data = KnowledgeBase()

        # get dicts from knowledge base
        self.environment_molecules_fw = self._get_environment_molecules_fw(raw_data)  # TODO (Eran) -- this assumes wcEcoli environment
        self.recipe_constructor = RecipeConstructor()
        self.recipes = self._get_recipes(raw_data)
        self._get_stock_media(raw_data)


    def _get_environment_molecules_fw(self, raw_data):
        '''get formula weight (units.g / units.mol) for all environmental molecules'''

        environment_molecules_fw = {}
        for row in raw_data.wcEcoli_environment_molecules:
            mol = row["molecule id"]
            fw = row["formula weight"]
            if fw == 'None':
                environment_molecules_fw[mol] = None
            else:
                environment_molecules_fw[mol] = float(fw) * (units.g / units.mol)

        return environment_molecules_fw

    def _get_stock_media(self, raw_data):
        '''load all stock media'''
        self.stock_media = {}

        for label in vars(raw_data.media):
            self.stock_media[label] = {}

            # get non-zero concentrations (assuming units.mmol / units.L)
            molecule_concentrations = getattr(raw_data.media, label)

            environment_non_zero_dict = {
                row["molecule id"]: row["concentration"]
                for row in molecule_concentrations}

            # update saved_media with non zero concentrations
            self.stock_media[label].update(environment_non_zero_dict)

        # add recipes to stock_media
        for media_id, recipe_raw in self.recipes.items():
            recipe_parsed = grammar.parse(recipe_raw)
            recipe = self.recipe_constructor.visit(recipe_parsed)
            media = self.make_recipe(recipe[0], True) # list only contains one recipe. Use first element.
            self.stock_media[media_id] = media

    def _get_recipes(self, raw_data):
        recipes = {}
        for row in raw_data.media_recipes:
            new_media_id = row["media id"]
            recipe = row["recipe"]
            recipes[new_media_id] = recipe
        return recipes

    def remove_units(self, media):
        return {mol: conc.m_as(CONC_UNITS) for mol, conc in media.items()}

    def get_saved_media(self, media_id, units=False):
        media = self.stock_media.get(media_id)
        if not units:
            media = self.remove_units(media)
        return media

    def add_media(self, media, new_media_id = str(uuid.uuid1())):
        '''add a media to stock'''

        # check/add units, if none defined, assume CONC_UNITS
        if all(isinstance(x, (float, int)) for x in media.values()):
            for mol_id, conc in media.items():
                media[mol_id] = conc * CONC_UNITS
        elif any(isinstance(x, (float, int)) for x in media.values()):
            raise AddIngredientsError(
                "inconsistent units for {}".format(new_media_id))

        self.stock_media[new_media_id] = media
        return new_media_id

    def make_recipe(self, recipe, units=False):
        '''make a single media recipe'''

        if recipe == 'end':
            return 'end'

        new_media = {}
        total_volume = 0.0 * VOLUME_UNITS
        for ingredient, amount in recipe.items():

            added_volume = amount.get('volume', 0.0 * VOLUME_UNITS)
            added_counts = amount.get('counts')
            added_weight = amount.get('weight')
            operation = amount.get('operation', 'add')

            added_media = {}
            if ingredient in self.stock_media:
                added_media = self.stock_media[ingredient]

            # if it is not an existing media, it needs weight or counts to add
            elif added_weight is not None:
                # added_weight takes priority over added_counts
                if added_weight.magnitude == INF:
                    added_conc = INF * CONC_UNITS
                elif added_weight.magnitude == 0.0:
                    added_conc = 0.0 * CONC_UNITS
                elif added_weight.magnitude < 0:
                    raise AddIngredientsError(
                        "Negative weight given for {}".format(ingredient))
                elif self.environment_molecules_fw[ingredient] is not None:
                    fw = self.environment_molecules_fw[ingredient]
                    added_counts = added_weight / fw
                    added_conc = added_counts / added_volume
                else:
                    raise AddIngredientsError(
                        "No fw defined for {} in wcEcoli_environment_molecules.tsv".format(ingredient))

                # save concentration
                added_media[ingredient] = added_conc

            elif added_counts is not None:
                # get new concentration
                # make infinite concentration of ingredient if mix_counts is Infinity
                if added_counts.magnitude == INF:
                    added_conc = INF * CONC_UNITS
                elif added_counts.magnitude == 0.0:
                    added_conc = 0.0 * CONC_UNITS
                elif added_counts.magnitude < 0:
                    raise AddIngredientsError(
                        "Negative counts given for {}".format(ingredient))
                else:
                    added_conc = added_counts / added_volume

                # save concentration
                added_media[ingredient] = added_conc

            else:
                raise AddIngredientsError(
                    "No added added weight or counts for {}".format(ingredient))

            if total_volume.magnitude == 0.0 and added_volume.magnitude == 0.0:
                # no volume, just merge media dicts. This is likely due to a call to a stock_media.
                new_media.update(added_media)
            else:
                new_media = self.combine_media(new_media, total_volume, added_media, added_volume, True, operation)

            total_volume += added_volume

        if not units:
            new_media = self.remove_units(new_media)

        return new_media

    def combine_media(self, media_1, volume_1, media_2, volume_2, units=False, operation='add'):

        # intialize new_media
        new_media = {mol_id: 0.0 * CONC_UNITS for mol_id in set(list(media_1.keys()) + list(media_2.keys()))}

        # get new_media volume
        new_volume = volume_1 + volume_2

        for mol_id in list(new_media.keys()):
            conc_1 = media_1.get(mol_id, 0 * CONC_UNITS)
            conc_2 = media_2.get(mol_id, 0 * CONC_UNITS)

            if conc_1.magnitude == INF or conc_2.magnitude == INF:
                new_conc = INF * CONC_UNITS
                if operation == 'subtract':
                    new_conc = 0.0 * CONC_UNITS
            else:
                counts_1 = conc_1 * volume_1
                counts_2 = conc_2 * volume_2
                new_counts = counts_1 + counts_2

                if operation == 'subtract':
                    new_counts = counts_1 - counts_2
                if new_counts.magnitude < 0:
                    raise AddIngredientsError(
                        "subtracting {} goes negative".format(mol_id))

                new_conc = new_counts / new_volume

            # update media
            new_media[mol_id] = new_conc

        if not units:
            new_media = self.remove_units(new_media)

        return new_media

    def make_timeline(self, timeline_str):
        '''
        Make a timeline from a string
        Args:
            timeline_str (str): 'time1 recipe1, time2 recipe2'
        Returns:
            timeline (list[tuple]): a list of tuples with (time (float), recipe (dict))
        '''

        timeline_parsed = grammar.parse(timeline_str)
        timeline_recipes = self.recipe_constructor.visit(timeline_parsed)
        timeline = []

        for time, recipe in timeline_recipes:
            media = self.make_recipe(recipe, True)
            if media == 'end':
                new_media_id = 'end'
            else:
                new_media_id = str(uuid.uuid1())
                # determine if this is an existing media
                for media_id, concentrations in self.stock_media.items():
                    if concentrations == media:
                    # if cmp(concentrations, media) == 0:
                        new_media_id = media_id
                        break
                self.stock_media[new_media_id] = media

            timeline.append((float(time), new_media_id))

        return timeline


class RecipeConstructor(NodeVisitor):
    '''
    Make a recipe from a parsed recipe expression.
    Args:
        - node: The node we're visiting
        - visited_children: The results of visiting the children of that node, in a list
    '''
    def visit_timeline(self, node, visited_children):
        timeline = []
        for child in visited_children:
            if child:
                timeline.append(child)
        return timeline

    def visit_one_event(self, node, visited_children):
        time, _, recipes, _ = visited_children

        if any(key in recipes for key in ['end', 'End', 'END']):
            event = (time[0], 'end')
        elif isinstance(time, list):
            event = (time[0], recipes)
        else:
            event = (recipes)
        return event

    def visit_recipe(self, node, visited_children):
        # put all added ingredients into a recipe dictionary
        recipe = {}
        for child in visited_children:
            recipe.update(child)
        return recipe

    def visit_one_ingredient(self, node, visited_children):
        add, subtract, ingredients = visited_children
        if isinstance(add, list):
            for amount in list(ingredients.values()):
                amount['operation'] = 'add'
        elif isinstance(subtract, list):
            for amount in list(ingredients.values()):
                amount['operation'] = 'subtract'
        return ingredients

    def visit_ingredient(self, node, visited_children):
        # put ingredients into small dicts, with their counts and volumes
        ingredient, counts, volume, infinity = visited_children
        if isinstance(counts, list) and isinstance(volume, list):
            recipe = {ingredient: {'counts': counts[0] * units.mmol, 'volume': volume[0] * units.L}}
        elif isinstance(volume, list):
            recipe = {ingredient: {'volume': volume[0] * units.L}}
        elif isinstance(infinity, list):
            recipe = {ingredient: {'counts': INF * units.L}}
        else:
            recipe = {ingredient: {}}
        return recipe

    def visit_counts(self, node, visited_children):
        return visited_children[1]

    def visit_volume(self, node, visited_children):
        return visited_children[1]

    def visit_media_id(self, node, visited_children):
        return (node.text)

    def visit_inf(self, node, visited_children):
        return INF

    def visit_numeric(self, node, visited_children):
        return float(node.text)

    def visit_add(self, node, visited_children):
        pass

    def visit_break(self, node, visited_children):
        pass

    def visit_ws(self, node, visited_children):
        pass

    def generic_visit(self, node, visited_children):
        # The generic visit method.
        return visited_children or node


def test_make_media():
    media_obj = Media()

    # Retrieve stock media
    media1 = media_obj.get_saved_media('M9_GLC', True)

    # Make media from ingredients
    ingredients = {
        'L-ALPHA-ALANINE': {'weight': 1.78 * units.g, 'volume': 0.025 * units.L},
        'ARG': {'weight': 8.44 * units.g, 'volume': 0.1 * units.L},
        'UREA': {'counts': 102.0 * units.mmol, 'volume': 1.0 * units.L},
        'LEU': {'weight': float("inf") * units.g, 'volume': 0 * units.L},
        'OXYGEN-MOLECULE': {'counts': 0 * units.g, 'volume': 0 * units.L},
    }
    media2 = media_obj.make_recipe(ingredients, True)

    # Combine two medias
    media3 = media_obj.combine_media(media1, 0.8 * units.L, media2, 0.2 * units.L)

    #Recipes with parsing expression grammer
    rc = RecipeConstructor()
    recipe_str = 'GLT 0.2 mmol 1 L + LEU 0.05 mmol .1 L + ARG 0.1 mmol .5 L'
    recipe_parsed = grammar.parse(recipe_str)
    recipe = rc.visit(recipe_parsed)
    media4 = media_obj.make_recipe(recipe[0])

    # Make timelines
    timeline_str = '0 minimal 1 L + GLT 0.2 mmol 1 L + LEU 0.05 mmol .1 L, ' \
                   '10 minimal_minus_oxygen 1 L + GLT 0.2 mmol 1 L, ' \
                   '100 minimal 1 L + GLT 0.2 mmol 1 L'
    timeline = media_obj.make_timeline(timeline_str)
    media1_id = timeline[0][1]
    media5 = media_obj.get_saved_media(media1_id)

if __name__ == '__main__':
    output = test_make_media()
