"""deeplay Application.fit builds DataLoader(train_data, batch_size=batch_size, shuffle=True).
(1) What happens if a 'BatchedDataset' returns a whole (B,C,H,W) batch per __getitem__ (as §10.2 L2 describes)?
(2) Does an IterableDataset survive shuffle=True?
(3) Does DataLoader hand the whole index list to __getitems__ (so one batched render per step is possible)?"""
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
print("torch", torch.__version__)

class ReturnsBatches(Dataset):          # literal reading of §10.2 L2: each item is a rendered batch of 64
    def __len__(self): return 1000
    def __getitem__(self, i): return torch.zeros(64, 1, 32, 32), torch.zeros(64, 2)
x, y = next(iter(DataLoader(ReturnsBatches(), batch_size=32, shuffle=True)))
print("(1) returns-batches dataset under deeplay's loader -> x", tuple(x.shape), "y", tuple(y.shape))

class Stream(IterableDataset):
    def __iter__(self):
        while True: yield torch.zeros(1, 32, 32), torch.zeros(2)
try:
    DataLoader(Stream(), batch_size=32, shuffle=True)
    print("(2) IterableDataset + shuffle=True -> accepted")
except ValueError as e:
    print("(2) IterableDataset + shuffle=True -> ValueError:", e)

calls = []
class Batched(Dataset):                 # proposed fix: per-item protocol + __getitems__ = one batched render
    def __len__(self): return 1000
    def __getitem__(self, i): raise AssertionError("per-item path should not be used")
    def __getitems__(self, idx):
        calls.append(len(idx))
        imgs = torch.stack([torch.full((1, 32, 32), float(i)) for i in idx])   # render(image_index=idx) in ONE call
        return [(imgs[k], torch.tensor([float(i), 0.])) for k, i in enumerate(idx)]
xb, yb = next(iter(DataLoader(Batched(), batch_size=32, shuffle=True)))
print("(3) __getitems__ called with", calls, "indices per call; batch x", tuple(xb.shape), "y", tuple(yb.shape),
      "; image i matches label i:", bool((xb[:, 0, 0, 0] == yb[:, 0]).all()))
