# Example use of make_media

### Create media object
> media_obj = Media()

### Retrieve stock media
Saved media from ```/environment/condition/media_recipes.tsv``` can be retrieved with the ```get_saved_media``` function:
> media1 = media_obj.get_saved_media('M9_GLC')

### Make media from ingredients
Ingredients is a dict with molecule ids as the keys.
Each ingredient's value is a dict with {'weight': value * (units.g), 'counts': value * (units.mmol), 'volume': value *  (units.L)}.
Only one of 'weights' (in units.g) or 'counts' (in units.mmol) is required; if both are specified, it will use weight.
If weight or counts is Infinity, it sets the final concentration to inf.
If weight or counts is 0, it sets the final concentration to 0.

> ingredients = {
	'L-ALPHA-ALANINE': {'weight': 1.78 * units.g, 'volume': 0.025 * units.L},
	'ARG': {'weight': 8.44 * units.g, 'volume': 0.1 * units.L},
	'UREA': {'counts': 102.0 * units.mmol, 'volume': 1.0 * units.L},
	'LEU': {'weight': float("inf") * units.g, 'volume': 0 * units.L},
	'OXYGEN-MOLECULE': {'counts': 0 * units.g, 'volume': 0 * units.L},
    }  
> media2 = media_obj.make_recipe(ingredients)

### Combine two medias
> media3 = media_obj.combine_media(media1, 0.8 * units.L, media2, 0.2 * units.L)

### Media with units
By default, ```get_saved_media```, ```make_recipe``` and ```combine_media``` return a media dictionary without units (unit=False).
To retrieve units, set units=True. 
> media1_units = media_obj.get_saved_media('M9_GLC', True)  
> media2_units = media_obj.make_recipe(ingredients, True)  
> media3_units = media_obj.combine_media(media1_units, 0.8 * units.L, media2_units, 0.2 * units.L, True)  

### Recipes with parsing expression grammer
Recipes can also be specified more simply with a string, and made into media with a parsing expression grammar:
> recipe_str = 'GLT 0.2 mmol 1 L + LEU 0.05 mmol .1 L + ARG 0.1 mmol .5 L'  
> recipe_parsed = grammar.parse(recipe_str)  

Make a recipe constructor object and pass the parsed recipe in. Note: use the first element of the recipe list
> rc = RecipeConstructor()  
> recipe = rc.visit(recipe_parsed)  
> media4 = media_obj.make_recipe(recipe[0])  


# Make timelines
Timelines are lists of events with times and media, with [(time1, media1_id), (time2, media2_id)].

### timeline strings
Timelines can be specified by a string, with events separated by commas, and each event given a time (in seconds) and media.

Timeline strings can specifying stock media:
> timeline_str = '0 minimal, 10 minimal_minus_oxygen, 100 minimal'

New media can combine existing media with + operation, specifying volumes of each:
> timeline_str = '0 M9_GLC 0.8 L + 5X_supplement_EZ 0.2 L, 100 M9_GLC 0.5 L + 5X_supplement_EZ 0.5 L'

New media can be specified by adding ingredients to existing media with +/- operations:
> timeline_str = '0 minimal 1 L + GLT 0.2 mmol 1 L + LEU 0.05 mmol .1 L, 10 minimal_minus_oxygen 1 L + GLT 0.2 mmol 1 L, 100 minimal 1 L + GLT 0.2 mmol 1 L' 

Ingredients can be removed from existing media, with Infinity removing all of the ingredient:
> timeline_str = '0 M9_GLC, 10 M9_GLC 1.0 L - GLC Infinity'

A simulation end time can also be specified in the timeline, by giving the time along with the word 'end':
> timeline_str = '0 M9_GLC, 1000 end'

### construct timelines
The timeline is constructed by passing the timeline string to the media object
> timeline = media_obj.make_timeline(timeline_str)

The media_ids in the timeline either use prespecified ids from the stock_media, or assigned a uuid. 
media_ids are saved to ```stock_media```, and can be retrieved:
> media1_id = timeline[0][1]
> media5 = media_obj.get_saved_media(media1_id)