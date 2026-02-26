import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.section = torch.tensor(data, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.section)

    def __getitem__(self, idx):
        section = self.section[idx].unsqueeze(0)
        return section

    def __str__(self):
        num_samples = len(self)
        return f"num_samples={num_samples}"
