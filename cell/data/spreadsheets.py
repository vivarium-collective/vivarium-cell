"""
Subclasses of DictWriter and DictReader that parse plaintext as JSON strings,
allowing for basic type parsing and fields that are dictionaries or lists.
"""

import os
import csv
import json
import re
import numpy as np

try:
    from future_builtins import filter
except ImportError:
    pass

from vivarium.library.units import units

TSV_DIALECT = csv.excel_tab

def array_to_list(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()

    return value


class JsonWriter(csv.DictWriter):
    def __init__(self, *args, **kwargs):
        csv.DictWriter.__init__(
            self, quotechar = "'", quoting = csv.QUOTE_MINIMAL, lineterminator="\n", *args, **kwargs
            )

    def _dict_to_list(self, rowdict):
        return csv.DictWriter._dict_to_list(self, {
            key:json.dumps(array_to_list(value))
            for key, value in rowdict.viewitems()
            })


def split_units(field):
    try:
        attribute = re.search(r'(.*?) \(', field).group(1)
        units_value =  eval(re.search(r'\((.*?)\)', field).group(1))
        return (attribute, units_value)
    except AttributeError:
        return (field, None)


class JsonReader(csv.DictReader):
    def __init__(self, *args, **kwargs):
        csv.DictReader.__init__(
            self,
            quotechar = "\"",
            quoting = csv.QUOTE_MINIMAL,
            *args, **kwargs)

        # This is a hack to strip extra quotes from the field names
        # Not proud of it, but it works.
        self.fieldnames # called for side effect

        self._fieldnames = [
            fieldname.strip('"') for fieldname in self._fieldnames]

        self.field_mapping = {
            field: split_units(field)
            for field in self._fieldnames}


def load_tsv(path):
    with open(path, 'rU') as tsvfile:
        reader = JsonReader(
            filter(lambda x: x.lstrip()[0] != "#", tsvfile),  # Strip comments
            dialect=TSV_DIALECT)
        attr_list = []
        for row in reader:
            entry = {}
            for field in reader.fieldnames:
                fieldname, units = reader.field_mapping[field]
                value = row[field]

                try:
                    value = json.loads(value)
                except:
                    pass

                if not units is None:
                    value *= units

                entry[fieldname] = value
            attr_list.append(entry)
    return attr_list
