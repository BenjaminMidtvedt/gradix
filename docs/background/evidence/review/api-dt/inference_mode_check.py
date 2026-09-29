"""§6.3: 'Data generation runs under torch.inference_mode() automatically when no resolved site requires gradients.'
§9(a) Stage 2 then feeds the generated batch straight into a network in training mode.
Check what happens with (i) inference_mode tensors, (ii) no_grad tensors, as network inputs and as loss targets."""
import torch, torch.nn as nn
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 2, 3, padding=1)).to(dev)

def simulate(mode):
    ctx = torch.inference_mode() if mode == "inference_mode" else torch.no_grad()
    with ctx:
        x = torch.rand(4, 1, 32, 32, device=dev) * 1000       # stand-in for sim output "x"
        heat = torch.rand(4, 1, 32, 32, device=dev)          # stand-in for label "heat"
    return {"x": x, "heat": heat}

for mode in ["inference_mode", "no_grad"]:
    b = simulate(mode)
    try:
        out = net(b["x"])
        loss = ((out[:, :1] - b["heat"]) ** 2).mean()
        loss.backward()
        print(f"{mode:15s}: training step OK")
    except RuntimeError as e:
        print(f"{mode:15s}: RuntimeError: {str(e).splitlines()[0]}")
    # in-place normalisation that users commonly do on a batch
    try:
        b["x"].sub_(b["x"].mean())
        print(f"{mode:15s}: in-place x.sub_() outside inference mode OK")
    except RuntimeError as e:
        print(f"{mode:15s}: in-place op RuntimeError: {str(e).splitlines()[0]}")
