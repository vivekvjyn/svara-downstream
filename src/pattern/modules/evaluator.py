import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalPrecision

from pattern.modules.dataset import Dataset

class Evaluator:
    def __init__(self, embedder, logger, device, window_size=100):
        self.embedder = embedder
        self.logger = logger
        self.window_size = window_size
        self.device = device

    def __call__(self, sequences, ids):
        similarity_matrix = self._similarity_matrix(sequences)

        map, mrr, p1, p5 = self._evaluate(similarity_matrix, ids)

        self.logger(f'MAP: {map:.8f}')
        self.logger(f'MRR: {mrr:.8f}')
        self.logger(f'Precision@1: {p1:.8f}')
        self.logger(f'Precision@5: {p5:.8f}')

        self._boxplot(similarity_matrix, ids)

        return map, mrr, p1, p5

    def _similarity_matrix(self, sequences):
        similarity_matrix = np.zeros((len(sequences), len(sequences)))

        for i in range(len(sequences)):
            self.logger(f"[{i + 1}/{len(sequences)}]")
            for j in range(len(sequences)):
                self.logger.pbar(j + 1, len(sequences))

                similarities = []
                for l in range(0, len(sequences) + self.window_size, self.window_size):
                    x = sequences[i][l:l+self.window_size]
                    y = sequences[j][l:l+self.window_size]

                    if not x.shape[0] or not y.shape[0]:
                        break

                    x_loader = torch.utils.data.DataLoader(Dataset((np.array([x])), self.device), batch_size=1, shuffle=False, num_workers=0)
                    y_loader = torch.utils.data.DataLoader(Dataset((np.array([y])), self.device), batch_size=1, shuffle=False, num_workers=0)
                    x_embedding = self.embedder(x_loader)
                    y_embedding = self.embedder(y_loader)
                    similarities.append(cosine_similarity(x_embedding, y_embedding)[0][0])

                similarity_matrix[i][j] = np.mean(similarities)

        return torch.tensor(similarity_matrix, dtype=torch.float32).to(self.device)

    def _evaluate(self, similarity_matrix, ids):
        preds = similarity_matrix.clone()
        preds[torch.eye(len(ids), dtype=bool)] = -torch.inf
        preds = preds.flatten()
        targets = (ids[:, None] == ids[None, :])
        targets[torch.eye(len(ids), dtype=bool)] = False
        targets = targets.flatten()
        indexes = torch.arange(len(ids)).to(self.device).repeat_interleave(len(ids))

        map = RetrievalMAP()(preds, targets, indexes)
        mrr = RetrievalMRR()(preds, targets, indexes)
        p1 = RetrievalPrecision(top_k=1)(preds, targets, indexes)
        p5 = RetrievalPrecision(top_k=5)(preds, targets, indexes)

        return map.item(), mrr.item(), p1.item(), p5.item()

    def _boxplot(self, similarity_matrix, ids):
        average_precisions = {torch.unique(ids)[i].item(): [] for i in range(len(torch.unique(ids)))}
        reciprocal_ranks = {torch.unique(ids)[i].item(): [] for i in range(len(torch.unique(ids)))}
        precision1 = {torch.unique(ids)[i].item(): [] for i in range(len(torch.unique(ids)))}
        precision5 = {torch.unique(ids)[i].item(): [] for i in range(len(torch.unique(ids)))}

        for i in range(similarity_matrix.shape[0]):
            preds = similarity_matrix[i].clone()
            preds[i] = -torch.inf
            targets = (ids == ids[i])
            targets[i] = False
            indexes = torch.zeros_like(ids).long()

            map = RetrievalMAP()(preds, targets, indexes)
            mrr = RetrievalMRR()(preds, targets, indexes)
            p1_score = RetrievalPrecision(top_k=1)(preds, targets, indexes)
            p5_score = RetrievalPrecision(top_k=5)(preds, targets, indexes)

            average_precisions[ids[i].item()].append(map.item())
            reciprocal_ranks[ids[i].item()].append(mrr.item())
            precision1[ids[i].item()].append(p1_score.item())
            precision5[ids[i].item()].append(p5_score.item())

        self._plot(average_precisions, "MAP")
        self._plot(reciprocal_ranks, "MRR")
        self._plot(precision1, "Precision@1")
        self._plot(precision5, "Precision@5")


    def _plot(self, data, metric):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[data[id] for id in sorted(data.keys())], palette="tab10")
        plt.xticks(ticks=range(len(data)), labels=sorted(data.keys()))
        plt.xlabel("Phrase ID")
        plt.ylabel(f"{metric}")
        plt.title(f"Phrase-wise {metric}")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        os.makedirs('results', exist_ok=True)
        plt.savefig(os.path.join('results', f"{metric.lower()}.png"))
        plt.close()
