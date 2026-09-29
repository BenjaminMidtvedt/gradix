"""Revision 4: optimiser state for per-image parameter tables.

See architecture.md §6.9, §9g and Appendix F.5.

A fit over a dataset gathers the rows of a per-image table (e.g. rough positions) batch by batch.
Dense Adam keeps moving rows that are absent from the current batch, through its momentum;
torch.optim.SparseAdam with a sparse embedding updates only the rows present.

Run from the repo root: uv run python docs/background/evidence/review/rev4/adam_rows.py
"""

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
rows, slots, lr = 10, 4, 1e-2
batches = ([0, 1, 2], [3, 4, 5], [3, 4, 5], [3, 4, 5])  # rows 0-2 are absent after the first batch

for sparse in (False, True):
    table = torch.nn.Embedding(rows, slots * 3, sparse=sparse).to(dev)
    opt_cls = torch.optim.SparseAdam if sparse else torch.optim.Adam
    opt = opt_cls(table.parameters(), lr=lr)
    after_first = None
    for idx in batches:
        # gather the batch's rows, then slice to its largest count (3 slots)
        pos = table(torch.tensor(idx, device=dev)).view(-1, slots, 3)[:, :3]
        loss = (pos**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if after_first is None:
            after_first = table.weight.detach()[:3].clone()
    drift = (table.weight.detach()[:3] - after_first).abs().max().item()
    name = "SparseAdam + sparse embedding" if sparse else "dense Adam"
    print(f"{name:30s}: rows 0-2 moved {drift:.2e} ({drift / lr:.2f} lr) "
          "over 3 steps without a gradient")
