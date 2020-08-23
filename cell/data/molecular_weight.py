# Note: proteins are referred to by their gene names with lower case letter
# molecular_weight is daltons (Da) from nucleotide sequence, taken from ecocyc

from vivarium.library.units import units

molecular_weight = {

    # protein monomers
    'flhD': 13316.0,
    'flhC': 21566.0,
    'fliL': 17221.0,
    'fliM': 37849.0,
    'fliN': 14855.0,
    'fliO': 12670.0,
    'fliP': 26928.0,
    'fliQ': 9632.0,
    'fliR': 28543.0,
    'fliE': 11127.0,
    'fliF': 60589.0,
    'fliG': 36776.0,
    'fliH': 25050.0,
    'fliI': 49316.0,
    'fliJ': 17307.0,
    'fliK': 39312.0,
    'flgA': 23519.0,
    'flgM': 10341.0,
    'flgN': 15867.0,
    'flgE': 42045.0,
    'flgB': 15240.0,
    'flgC': 13968.0,
    'flgD': 23575.0,
    'flgF': 25912.0,
    'flgG': 27744.0,
    'flgH': 24615.0,
    'flgI': 38169.0,
    'flgJ': 34475.0,
    'flhB': 42238.0,
    'flhA': 74843.0,
    'flhE': 14059.0,
    'fliA': 27521.0,
    'fliZ': 21658.0,
    'fliD': 48456.0,
    'fliS': 14950.0,
    'fliT': 13829.0,
    'flgK': 57930.0,
    'flgL': 34281.0,
    'fliC': 51295.0,
    'tar': 59944.0,
    'tap': 57512.0,
    'cheR': 32849.0,
    'cheB': 37468.0,
    'cheY': 14097.0,
    'cheZ': 23976.0,
    'motA': 32011.0,
    'motB': 34186.0,
    'cheA': 71382.0,
    'cheW': 18084.0,

    # complexes
    'flhDC': 96396.0,  # from stoichiometry
    'flagellar export apparatus': 808657.0,  # from stoichiometry
    'flagellar motor switch': 2257897.0,   # from stoichiometry
    'flagellar motor': 2575900.0,  # from stoichiometry
    'flagella': 8815743.0,  # from stoichiometry
}

molecular_weight = {
    key: value * units.g / units.mol
    for key, value in molecular_weight.items()}
