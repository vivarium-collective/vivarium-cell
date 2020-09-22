import os
import csv

from vivarium_cell.data.spreadsheets import load_tsv

FLAT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "flat")

LIST_OF_FLAT_FILENAMES = (
    os.path.join("wcEcoli_genes.tsv"),
    os.path.join("wcEcoli_proteins.tsv"),
    os.path.join("wcEcoli_environment_molecules.tsv"),
    os.path.join("timelines_def.tsv"),
    os.path.join("media_recipes.tsv"),
    os.path.join("media", "wcEcoli_base.tsv"),
    os.path.join("media", "M9.tsv"),
    os.path.join("media", "M9_GLC.tsv"),
    os.path.join("media", "5X_supplement_EZ.tsv"),
    os.path.join("media", "GLC_G6P.tsv"),
    os.path.join("media", "GLC_LCT.tsv"),
    os.path.join("media", "ecoli_core_GLC.tsv"),
    os.path.join("media", "PURE_Fuji_2014.tsv"),
    os.path.join("media", "PURE_Ueda_2010.tsv"),
)

class DataStore(object):
    def __init__(self):
        pass

class KnowledgeBase(object):
    """ KnowledgeBase """

    def __init__(self):
        # Load raw data from TSV files

        for filename in LIST_OF_FLAT_FILENAMES:
            self._load_tsv(FLAT_DIR, filename)

        self.genes = {
            gene['symbol']: gene
            for gene in self.wcEcoli_genes}

        self.proteins = {
            protein['geneId']: protein
            for protein in self.wcEcoli_proteins}

    def _load_tsv(self, dir_name, file_name):
        path = self
        steps = file_name.split(os.path.sep)
        for subPath in steps[:-1]:
            if not hasattr(path, subPath):
                setattr(path, subPath, DataStore())
            path = getattr(path, subPath)
        attrName = steps[-1].split(".")[0]
        setattr(path, attrName, [])

        file_path = os.path.join(dir_name, file_name)
        rows = load_tsv(file_path)
        setattr(path, attrName, [row for row in rows])

    def concatenate_sequences(self, units):
        sequence = ''
        for unit in units:
            gene = self.genes[unit]
            protein = self.proteins[gene['id']]
            sequence += protein['seq']
        return sequence

