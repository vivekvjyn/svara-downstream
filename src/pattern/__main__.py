import os
import argparse
import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

from pattern import Model, Evaluator, Embedder, Logger, Dataset, Table, load_pitch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger()
    args = parse_args()

    # Load dataset
    sections = load_pitch(os.path.join("dataset", "segments.pkl"))
    with open(os.path.join("dataset", "ids.pkl"), "rb") as f:
        ids = pickle.load(f)

    # get count of each id
    id_counts = {}
    for id in ids:
        if id not in id_counts:
            id_counts[id] = 0
        id_counts[id] += 1

    model = Model(embed_dim=args.embed_dim, num_classes=len(set(ids)), depth=args.depth).to(device)
    model.encoder.load(os.path.join("checkpoints", "encoder.pth"), device)
    embedder = Embedder(model, logger)

    map = []
    mrr = []
    mean_precision = []
    for i in range(len(sections)):
        distances = np.zeros((len(sections)))
        for j in range(len(sections)):
            logger.pbar(j + 1, len(sections))
            distance = 0

            window_size = 200

            for l in range(0, len(sections) + window_size, window_size):
                window_i = sections[i][l:l+window_size]
                window_j = sections[j][l:l+window_size]

                if not window_i.shape[0] and window_j.shape[0]:
                    break

                data_loader_i = torch.utils.data.DataLoader(Dataset((np.array([window_i])), device), batch_size=args.batch_size, shuffle=False, num_workers=0)
                data_loader_j = torch.utils.data.DataLoader(Dataset((np.array([window_j])), device), batch_size=args.batch_size, shuffle=False, num_workers=0)
                embedding_i = embedder(data_loader_i)
                embedding_j = embedder(data_loader_j)
                distance += np.linalg.norm(embedding_i - embedding_j)
                # cosine
                #distance += cosine(embedding_i.flatten(), embedding_j.flatten())

            distances[j] = distance

        ranking = np.argsort(distances)
        ranking = ranking[ranking != i]
        ranked_ids = np.array(ids)[ranking]
        positives = np.where(ranked_ids == ids[i])[0]

        # reciprocal rank
        rr = 1 / (positives[0] + 1)

        precision_at_10 = np.sum(positives < 10) / 10

        precisions = []
        for rank_idx in positives:
            precision_at_rank = np.sum(positives <= rank_idx) / (rank_idx + 1)
            precisions.append(precision_at_rank)

        ap = np.mean(precisions)

        map.append(ap)
        mrr.append(rr)
        mean_precision.append(precision_at_10)

        print(f"\nAP for section {i} ID {ids[i]}: {ap}")
        print(f"Reciprocal Rank for section {i} ID {ids[i]}: {rr}")
        print(f"Precision@10 for section {i} ID {ids[i]}: {precision_at_10}")

    map = np.mean(map)
    mrr = np.mean(mrr)
    mean_precision = np.mean(precision_at_10)
    print(f"MAP: {map}")
    print(f"MRR: {mrr}")
    print(f"Mean Precision@10: {mean_precision}")




def parse_args():
    parser = argparse.ArgumentParser(description="svara representation learning for carnatic music transcription")
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--catchup', type=int, default=10, help='number of epochs to freeze encoders (default: 10)')
    parser.add_argument('--early-stopping', type=int, default=10, help='early stopping patience (default: 30)')
    parser.add_argument('--embed-dim', type=int, default=48, help='dimension of embedding space (default: 48)')
    parser.add_argument('--depth', type=int, default=5, help='number of inception modules (default: 5)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='weight decay (default: 1e-3)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
