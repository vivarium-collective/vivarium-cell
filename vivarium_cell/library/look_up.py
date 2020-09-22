from __future__ import absolute_import, division, print_function

import os
import csv
import random
from vivarium_cell.data.spreadsheets import load_tsv

FLAT_DIR = os.path.join('vivarium', 'data', 'flat')
MEDIA_IDS = ['minimal', 'minimal_minus_oxygen', 'minimal_plus_amino_acids']
FLUX_STR = 'lookup_flux_'

class LookUp(object):

    def __init__(self):

        self.lookup_avg = {media_id: {} for media_id in MEDIA_IDS}
        self.lookup_dist = {media_id: {} for media_id in MEDIA_IDS}

        for media_id in MEDIA_IDS:
            filename=os.path.join(FLAT_DIR, FLUX_STR + media_id + '.tsv')
            avg, dist = load_lookup(filename)
            self.lookup_avg[media_id].update(avg)
            self.lookup_dist[media_id].update(dist)

    def look_up(self, lookup_type, media, keys):
        ''' return look up values for each key in keys'''

        values = {}
        if lookup_type == 'average':
            values = {
                key: self.lookup_avg[media].get(key, None)
                for key in keys}
        if lookup_type == 'distribution':
            values = {
                key: random.choice(self.lookup_avg[media].get(key, None))
                for key in keys}

        return values


def load_lookup(filename):
    ''' load a file and pass back dicts with lookup_avg and lookup_dist'''
    lookup_avg = {}
    lookup_dist = {}

    rows = load_tsv(filename)
    for row in rows:
        key = row.get('id')
        avg = row.get('average')
        dist = row.get('distribution')

        # check if empty distribution
        if dist:
            dist = [float(value) for value in dist]
        else:
            dist=[0]

        lookup_avg[key] = avg
        lookup_dist[key] = dist

    return lookup_avg, lookup_dist
