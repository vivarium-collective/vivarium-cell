def invert_index(mapping):
    inverse = {}
    for canon, synonyms in mapping.items():
        for synonym in synonyms:
            inverse[synonym] = canon
    return inverse

synonyms = {
    # 'ATP': ['atp_c'],
    'ADP': ['adp_c'],
    'GTP': ['gtp_c'],
    'NAD(H)': ['nad_c'],
    'NADP(H)': ['nadp_c'],
    'FAD(H2)': ['fad_c'],
    'ATP': ['A', 'ATP', 'atp_c'],  # ATP name conflicts with the energy carrier
    'GTP': ['G', 'GTP', 'gtp_c'],
    'UTP': ['U', 'UTP', 'utp_c'],
    'CTP': ['C', 'CTP', 'ctp_c'],
    'Alanine': ['ala__L_c'],
    'Arginine': ['arg__L_c'],
    'Asparagine': ['asn__L_c'],
    'Aspartate': ['asp__L_c'],
    'Cysteine': ['cys__L_c'],
    'Glutamate': ['glu__L_c'],
    'Glutamine': ['gln__L_c'],
    'Glycine': ['gly_c'],
    'Histidine': ['his__L_c'],
    'Isoleucine': ['ile__L_c'],
    'Leucine': ['leu__L_c'],
    'Lysine': ['lys__L_c'],
    'Methionine': ['met__L_c'],
    'Phenylalanine': ['phe__L_c'],
    'Proline': ['pro__L_c'],
    'Serine': ['ser__L_c'],
    'Threonine': ['thr__L_c'],
    'Tryptophan': ['trp__L_c'],
    'Tyrosine': ['tyr__L_c'],
    'Valine': ['val__L_c'],
}

index = invert_index(synonyms)

def get_synonym(id):
    return index.get(id, id)

