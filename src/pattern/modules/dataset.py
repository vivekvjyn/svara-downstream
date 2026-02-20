import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.section = torch.tensor(data[0], dtype=torch.float32).to(device)
        self.targets = torch.tensor(data[1], dtype=torch.long).to(device)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        section = self.section[idx].unsqueeze(0)
        targets = self.targets[idx]
        return section, targets

    def __str__(self):
        num_samples = len(self)
        class_counts = torch.bincount(self.targets).cpu().numpy()
        class_distribution = {i: count for i, count in enumerate(class_counts)}
        return f"num_samples={num_samples}, num_class={self.num_class} distribution={class_distribution}"

    @property
    def num_class(self):
        return len(torch.unique(self.targets))
