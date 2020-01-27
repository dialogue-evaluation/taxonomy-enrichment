from bs4 import BeautifulSoup
import os
import codecs

from database import DatabaseRuWordnet


def get_soup(file):
    with codecs.open(file, encoding='utf-8')as f:
        handler = f.read()
    return BeautifulSoup(handler, features="lxml")


def parse_synsets(file):
    soup = get_soup(file)
    return [(element.attrs['id'], element.attrs['ruthes_name']) for element in soup.findAll('synset')]


def parse_relations(file):
    relations = []
    soup = get_soup(file)
    for element in soup.findAll('relation'):
        relation = element.attrs
        if relation['name'] in ('hypernym', 'instance hypernym'):
            relations.append((relation['parent_id'], relation['child_id']))
    return relations


def parse_senses_lemmas(file):
    soup = get_soup(file)
    return [(el.attrs['id'], element.attrs['id'], el.text) for element in soup.findAll('synset')
           for el in element.findAll('sense')]


def parse_senses(file):
    soup = get_soup(file)
    return [(element.attrs['id'], element.attrs['synset_id'], element.attrs['name']) for element in
            soup.findAll('sense')]


def get_wordnet_files_from_path(path):
    synsets = []
    relations = []
    senses = []
    for directory, _, files in os.walk(path):
        for i in files:
            if i.startswith('synsets'):
                synsets.append(os.path.join(directory, i))
            elif i.startswith('synset_relation'):
                relations.append(os.path.join(directory, i))
            elif i.startswith('senses'):
                senses.append(os.path.join(directory, i))
    return synsets, relations, senses


class RuWordnet(DatabaseRuWordnet):
    def __init__(self, db_path, ruwordnet_path, with_lemmas=False):
        super(RuWordnet, self).__init__(db_path)
        self.__initialize_db(ruwordnet_path)
        self.with_lemmas = with_lemmas

    def __initialize_db(self, path):
        if self.is_empty():
            print("Inserting data to database")
            synset_files, relation_files, senses_files = get_wordnet_files_from_path(path)

            synsets = [synset for file in synset_files for synset in parse_synsets(file)]
            relations = [relation for file in relation_files for relation in parse_relations(file)]

            if self.with_lemmas:
                senses = [sense for file in synset_files for sense in parse_senses_lemmas(file)]
            else:
                senses = [sense for file in senses_files for sense in parse_senses(file)]

            self.insert_synsets(synsets)
            self.insert_relations(relations)
            self.insert_senses(senses)
