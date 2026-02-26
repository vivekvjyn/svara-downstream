import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

class Evaluator:
    def __init__(self, embedder, logger, window_size=200, timestep=0.0100006335037896):
        self.embedder = embedder
        self.timestep = timestep
        self.logger = logger
        self.window_size = window_size

    def __call__(self, sections, ids, k=10):
        distance_matrix = self._distance_matrix(sections)

        map = self._mean_average_precision(distance_matrix, ids)
        mrr = self._mean_reciprocal_rank(distance_matrix, ids)
        precision_at_k = self._precision_at_k(distance_matrix, ids, k=k)

        self.logger(f'MAP: {map:.8f}')
        self.logger(f'MRR: {mrr:.8f}')
        self.logger(f'Precision@{k}: {precision_at_k:.8f}')

        return map, mrr, precision_at_k


    def _distance_matrix(sections):
        distance_matrix = np.zeros((len(sections), len(sections)))

        for i in range(len(sections)):
            for j in range(len(sections)):
                logger.pbar(j + 1, len(sections))

                distance = 0
                for l in range(0, len(sections) + window_size, window_size):
                    x = sections[i][l:l+window_size]
                    y = sections[j][l:l+window_size]

                    if not x.shape[0] or not y.shape[0]:
                        break

                    x_loader = torch.utils.data.DataLoader(Dataset((np.array([x])), device), batch_size=args.batch_size, shuffle=False, num_workers=0)
                    y_loader = torch.utils.data.DataLoader(Dataset((np.array([y])), device), batch_size=args.batch_size, shuffle=False, num_workers=0)
                    x_embedding = embedder(x_loader)
                    y_embedding = embedder(y_loader)
                    distance += np.linalg.norm(x_embedding - y_embedding)

                distance_matrix[i, j] = distance

        return distance_matrix

    def _mean_average_precision(self, distance_matrix, ids):
        map = []
        for i in range(len(sections)):
            ranking = np.argsort(distance_matrix[i])
            ranking = ranking[ranking != i]
            ranked_ids = np.array(ids)[ranking]
            positives = np.where(ranked_ids == ids[i])[0]

            precisions = []
            for rank_idx in positives:
                precision_at_rank = np.sum(positives <= rank_idx) / (rank_idx + 1)
                precisions.append(precision_at_rank)

            ap = np.mean(precisions)
            map.append(ap)

        return np.mean(map)

    def _mean_reciprocal_rank(self, distance_matrix, ids):
        mrr = []
        for i in range(len(sections)):
            ranking = np.argsort(distance_matrix[i])
            ranking = ranking[ranking != i]
            ranked_ids = np.array(ids)[ranking]
            positives = np.where(ranked_ids == ids[i])[0]

            rr = 1 / (positives[0] + 1)
            mrr.append(rr)

        return np.mean(mrr)

    def _precision_at_k(self, distance_matrix, ids, k=10):
        precision_at_k = []
        for i in range(len(sections)):
            ranking = np.argsort(distance_matrix[i])
            ranking = ranking[ranking != i]
            ranked_ids = np.array(ids)[ranking]
            positives = np.where(ranked_ids == ids[i])[0]

            precision_at_k.append(np.sum(positives < k) / k)

        return np.mean(precision_at_k)
