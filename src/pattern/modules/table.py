import os
import pandas as pd

class Table:
    def __init__(self):
        self.dir = os.path.join('results')
        os.makedirs(self.dir, exist_ok=True)
        self.filename = 'pattern_ranks.tsv'

    def insert(self, map, mrr, precision_at_k):
        df = pd.DataFrame({
            'map': [map],
            'mrr': [mrr],
            'p@k': [precision_at_k]
        })

        df.to_csv(os.path.join(self.dir, self.filename), sep='\t', index=False)
