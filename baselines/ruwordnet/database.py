import sqlite3


class DatabaseRuWordnet(object):
    def __init__(self, path="ruwordnet/ruwordnet.db"):
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('PRAGMA encoding = "UTF-8"')
        self.cursor.execute('PRAGMA auto_vacuum = 1')
        self.create_ruwordnet()

    def is_empty(self):
        return self.cursor.execute('SELECT COUNT(id) FROM synsets').fetchall()[0][0] == \
               self.cursor.execute('SELECT COUNT(hypernym_id) FROM relations').fetchall()[0][0] == 0

    def create_ruwordnet(self):
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS synsets 
        (id text NOT NULL PRIMARY KEY, ruthes_name text)""")
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS senses 
                (sense_id text NOT NULL PRIMARY KEY, synset_id text, sense_name text)""")
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS relations 
        (hypernym_id text NOT NULL, hyponym_id text NOT NULL, PRIMARY KEY(hypernym_id, hyponym_id))""")
        self.conn.commit()

    def insert_synsets(self, synsets):
        self.cursor.executemany("INSERT INTO synsets VALUES (?,?)", synsets)
        self.conn.commit()

    def insert_relations(self, relations):
        self.cursor.executemany("INSERT INTO relations VALUES (?,?)", relations)
        self.conn.commit()

    def insert_senses(self, senses):
        self.cursor.executemany("INSERT INTO senses VALUES (?,?,?)", senses)
        self.conn.commit()

    def get_synset_names(self):
        return set([i[0] for i in self.cursor.execute('''SELECT ruthes_name FROM synsets''').fetchall()])

    def get_id_by_name(self, name):
        synset_id = self.cursor.execute('''SELECT id FROM synsets WHERE ruthes_name="{}"'''
                                        .format(name.upper())).fetchall()
        return synset_id[-1][0] if len(synset_id) > 0 else ''

    def get_name_by_id(self, synset_id):
        name = self.cursor.execute('''SELECT ruthes_name FROM synsets WHERE id="{}"'''.format(synset_id)).fetchall()
        return name[0][0] if len(name) > 0 else ''

    def get_hyponyms_by_name(self, name):
        synset_id = self.get_id_by_name(name)
        return self.get_hyponyms_by_id(synset_id)

    def get_hyponyms_by_id(self, synset_id):
        hyponym_list = self.cursor.execute('''SELECT hypernym_id FROM relations WHERE hyponym_id="{}"'''
                                           .format(synset_id))
        return [i[0] for i in hyponym_list.fetchall()]

    def get_hypernyms_by_name(self, name):
        synset_id = self.get_id_by_name(name)
        return self.get_hypernyms_by_id(synset_id)

    def get_hypernyms_by_id(self, synset_id):
        hypernym_list = self.cursor.execute('''SELECT hyponym_id FROM relations WHERE hypernym_id="{}"'''
                                            .format(synset_id))
        return [i[0] for i in hypernym_list.fetchall()]

    def get_all_relations(self):
        return self.cursor.execute('''SELECT * FROM relations''').fetchall()

    def get_all_synsets(self, endswith=""):
        return [i for i in self.cursor.execute('''SELECT * FROM synsets''').fetchall() if i[0].endswith(endswith)]

    def get_all_ids(self, endswith=""):
        return [i[0] for i in self.cursor.execute('''SELECT id FROM synsets''').fetchall() if i[0].endswith(endswith)]

    def get_all_senses(self):
        return self.cursor.execute('''SELECT * FROM senses''').fetchall()

    def is_hyponym(self, first, second):
        return self.cursor.execute('''SELECT * FROM relations WHERE hypernym_id="{}" and hyponym_id="{}"'''
                                            .format(first, second)).fetchall()

    def get_synset_by_sense(self, sense):
        synset_id = self.cursor.execute('''SELECT synset_id from senses where sense_name="{}"'''
                                        .format(sense)).fetchall()
        return synset_id[0][0] if len(synset_id) > 0 else ''
