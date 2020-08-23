#: List of dictionaries, one for each amino acid. Each dictionary has
#: keys ``name``, ``abbreviation``, and ``symbol``.
amino_acid_records = [
    {'name': 'Alanine', 'abbreviation': 'Ala', 'symbol': 'A'},
    {'name': 'Arginine', 'abbreviation': 'Arg', 'symbol': 'R'},
    {'name': 'Asparagine', 'abbreviation': 'Asn', 'symbol': 'N'},
    {'name': 'Aspartate', 'abbreviation': 'Asp', 'symbol': 'D'},
    {'name': 'Cysteine', 'abbreviation': 'Cys', 'symbol': 'C'},
    {'name': 'Glutamate', 'abbreviation': 'Glu', 'symbol': 'E'},
    {'name': 'Glutamine', 'abbreviation': 'Gln', 'symbol': 'Q'},
    {'name': 'Glycine', 'abbreviation': 'Gly', 'symbol': 'G'},
    {'name': 'Histidine', 'abbreviation': 'His', 'symbol': 'H'},
    {'name': 'Isoleucine', 'abbreviation': 'Ile', 'symbol': 'I'},
    {'name': 'Leucine', 'abbreviation': 'Leu', 'symbol': 'L'},
    {'name': 'Lysine', 'abbreviation': 'Lys', 'symbol': 'K'},
    {'name': 'Methionine', 'abbreviation': 'Met', 'symbol': 'M'},
    {'name': 'Phenylalanine', 'abbreviation': 'Phe', 'symbol': 'F'},
    {'name': 'Proline', 'abbreviation': 'Pro', 'symbol': 'P'},
    {'name': 'Serine', 'abbreviation': 'Ser', 'symbol': 'S'},
    {'name': 'Threonine', 'abbreviation': 'Thr', 'symbol': 'T'},
    {'name': 'Tryptophan', 'abbreviation': 'Trp', 'symbol': 'W'},
    {'name': 'Tyrosine', 'abbreviation': 'Tyr', 'symbol': 'Y'},
    {'name': 'Valine', 'abbreviation': 'Val', 'symbol': 'V'}]

#: Map from amino acid symbol to the amino acid name
amino_acids = {
    record['symbol']: record['name']
    for record in amino_acid_records}
