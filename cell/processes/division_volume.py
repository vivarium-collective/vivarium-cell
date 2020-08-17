from __future__ import absolute_import, division, print_function

from vivarium.library.units import units
from vivarium.core.process import Process



class DivisionVolume(Process):

    name = 'division_volume'
    defaults = {
        'initial_state': {},
        'division_volume': 2.4 * units.fL,  # fL
    }

    def __init__(self, initial_parameters=None):
        if not initial_parameters:
            initial_parameters = {}

        self.division = 0
        division_volume = initial_parameters.get('division_volume', self.defaults['division_volume'])

        parameters = {
            'division_volume': division_volume}  # TODO -- make division at 2X initial_volume?  Pass this in from initial_parameters

        super(DivisionVolume, self).__init__(parameters)

    def ports_schema(self):
        return {
            'global': {
                'divide': {
                    '_default': False,
                    '_emit': True,
                    '_updater': 'set',
                    '_divider': 'zero'},
                'volume': {
                    '_default': 1.2 * units.fL}}}

    def next_update(self, timestep, states):
        volume = states['global']['volume']
        if volume >= self.parameters['division_volume']:
            self.division = True
            return {'global': {'divide': self.division}}
        else:
            return {}
