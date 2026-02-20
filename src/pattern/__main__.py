import os
import argparse
import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from pattern import Model, Evaluator, Embedder, Logger, Dataset, Table, load_pitch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger()
    args = parse_args()

    # Load dataset
    sections, sections_processed = load_pitch(os.path.join("dataset", "segments.pkl"))
    with open(os.path.join("dataset", "ids.pkl"), "rb") as f:
        ids = pickle.load(f)

    dataset = Dataset((sections_processed, ids), device)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = Model(embed_dim=args.embed_dim, num_classes=len(set(ids)), depth=args.depth).to(device)
    model.encoder.load(os.path.join("checkpoints", "encoder.pth"), device)
    embedder = Embedder(model, logger)
    evaluator = Evaluator(logger)
    embeddings = embedder(data_loader)

    distances = np.zeros((len(embeddings)))
    query_embedding = embeddings[0]
    query_id = ids[0]
    for i, embedding in enumerate(embeddings):
        distance = np.linalg.norm(query_embedding - embedding)
        distances[i] = distance

        print(f"Query ID: {query_id}, Target ID: {ids[i]}, Distance: {distance:.4f}")



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
