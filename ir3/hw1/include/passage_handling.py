from scipy.sparse import csr_matrix
from helpers import inversion_count, factorial

class Handler:
    
    def __init__(self, min_size=1, max_size=10):
        self.min_size = min_size
        self.max_size = max_size
        discount = lambda size: size

    def __call__(self, body, query):
        query_set = set(query)
        doc_set = set(body)
        query_size = len(query_set)
        positions = csr_matrix((1, doc_len), dtype=np.int64)
        for pos, word in enumerate(doc):
            positions[pos] = max(query.find(word), 0)

        for start_position in range(len(body)):
            for passage_size in range(min_size, max_size):
                if len(body) - start_position > passage_size:
                    passage = body[start_position:start_position + passage_size]
                    # TF-IDF
                    # Полнота, порядок, правильность словоформ, кучность, близость к началу, зона документа
                    # if tfid < threshold:
                    tfidf = tfidf(passage, query)
                    fullness = len(query_set - doc_set) / query_size
                    close_to_start = start_position / len(body)
                    zone = passage.zone
                    inversions = inversions_count([x for x in positions[0, start_position:passage_size].tolist() if x >
                    0]) / factorial(query_len)

                    kuchnost = positions[0, start_position:passage_size].nonzero()[1]
                    kuchnost = query.size / kuchnost[-1] - kuchnost[0]
                
