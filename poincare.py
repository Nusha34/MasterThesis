from gensim.models.poincare import PoincareModel
from datetime import datetime
import pandas as pd
import sys

def hello_world():
    relations = [('math', 'science'), ('cs', 'science'), ('ml', 'cs'), ('db', 'cs'), ('linalg', 'math')]
    model = PoincareModel(relations, size=8, negative=2)
    model.train(epochs=50)
    # Poincare distance between two entities
    print(model.kv.distance('ml', 'db'))
    # Compute absolute position in hierarchy of input node or vector. 
    # Values range between 0 and 1. A lower value indicates the input 
    # node or vector is higher in the hierarchy.
    print(model.kv.norm('ml'))
    print(model.kv.norm('ml'))
    # Get the vectors
    print(model.kv.get_vector('ml'))
    model.save('test_embeddings.bin')
    model.kv.save_word2vec_format('test_embeddings.w2v')
    return

def load_snomed_isa_relations(path):
    print('Parsing %s' % path)
    relations = []
    t_start = datetime.now()
    isa_relations = pd.read_csv(path, delimiter='|', usecols=['SCUI1', 'SCUI2']) 
    for row_id, row in isa_relations.iterrows():
        relations.append((row['SCUI2'], row['SCUI1']))
    return relations

def get_poincare_model(relations, emb_size, num_threads=1):
    print('Learning Poincare embeddings with %d relations' % len(relations))
    model = PoincareModel(relations, size=emb_size, negative=2)
    t_start = datetime.now()
    model.train(epochs=50)
    t_end = datetime.now()
    print('Training time: %s' % (t_end - t_start))
    return model

def test_poincare(emb_size, num_threads, store_embeddings=False, run_test=False):
    path = '/projects/deepcare/deepcare/is-a-relations/SNOMEDCT_isa.txt'
    snomed_isa_relations = load_snomed_isa_relations(path)
    poincare_model = get_poincare_model(snomed_isa_relations, emb_size, num_threads)
    if run_test:
        print('Running tests ...')
        d1 = poincare_model.kv.distance('Suicide', 'Injury due to suicide attempt')
        d2 = poincare_model.kv.distance('Suicide', 'Diabetic macular edema')
        assert(d1 < d2)
        general = 'Suicide'
        specific = 'Suicide or selfinflicted injury by shotgun'
        assert(poincare_model.kv.norm(general) < poincare_model.kv.norm(specific))
    
    out_path = '%s.emb_dims_%d.nthreads_%d.txt' % (path, emb_size, num_threads)
    if store_embeddings:
        print('Saving embeddings to %s' % out_path)
        poincare_model.kv.save_word2vec_format(out_path)
    return

def gen_poincare_embeddings():
    #dimensions = [2, 5, 10, 20, 50, 100, 200]
    dimensions = [100, 200]
    num_threads = [1] #[1, 4, 8, 16, 22]
    for d in dimensions:
        for n_t in num_threads:
            print('***********************************************************')
            print('Generating embeddings with %d dimensions using %d threads' % (d, n_t))
            print('***********************************************************')
            test_poincare(d, n_t, store_embeddings=True)
    print('Done!')

if __name__ == '__main__':
    #hello_world()
    #test_poincare()
    gen_poincare_embeddings()
    sys.exit(0) 
