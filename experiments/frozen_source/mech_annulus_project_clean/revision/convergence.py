"""MECH convergence check: per-epoch training loss + validation probe metric.

Trains the searched configuration with per-epoch loss recording and evaluates
a validation-side loss (reconstruction + contrastive on validation queries)
to decide whether 24 epochs converged or validation-based early stopping at
up to 100 epochs is justified. Validation only; no test data.
"""
from __future__ import annotations
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import core as c
import run as r


def probe_loss(model, x_val, alpha):
    with torch.no_grad():
        hs = model(x_val)
        x_norm = F.normalize(x_val, dim=1)
        sim_x = x_norm @ x_norm.T
        rec = contrast = 0.0
        for h in hs:
            rec = rec + F.mse_loss(model.decoder(h), x_val)
            h_norm = F.normalize(h, dim=1)
            contrast = contrast + F.mse_loss(h_norm @ h_norm.T, sim_x)
        return float(rec / len(hs)), float(contrast / len(hs))


def run(name, tables, bits, max_epochs=100, probe_every=4, batch=128):
    arrays, split, meta, _, _ = r.setup(name)
    torch.set_num_threads(8)
    torch.manual_seed(r.MODEL_SEED)
    np.random.seed(r.MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    model = c.source.MultiEncoderContrastiveHash(arrays['train'].shape[1], tables, bits)
    opt = torch.optim.Adam(model.parameters(), lr=r.TRAIN['lr'])
    loader = DataLoader(TensorDataset(torch.from_numpy(arrays['train'])), batch_size=batch,
                        shuffle=True, drop_last=False)
    x_val = torch.from_numpy(arrays['validation'])
    curve = []
    step = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for (batch_x,) in loader:
            hs = model(batch_x)
            x_norm = F.normalize(batch_x, dim=1)
            sim_x = x_norm @ x_norm.T
            rec = contrast = quant = balance = decorr = 0.0
            for h in hs:
                rec = rec + F.mse_loss(model.decoder(h), batch_x)
                h_norm = F.normalize(h, dim=1)
                contrast = contrast + F.mse_loss(h_norm @ h_norm.T, sim_x)
                quant = quant + torch.mean((torch.abs(h) - 1.0) ** 2)
                balance = balance + torch.mean(torch.mean(h, dim=0) ** 2)
                corr = (h.T @ h) / max(h.shape[0], 1)
                off = corr - torch.diag(torch.diag(corr))
                decorr = decorr + torch.mean(off ** 2)
            loss = (rec / len(hs) + r.TRAIN['alpha'] * contrast / len(hs) +
                    r.TRAIN['beta'] * quant / len(hs) + r.TRAIN['gamma_balance'] * balance / len(hs) +
                    r.TRAIN['gamma_decorrelation'] * decorr / len(hs))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        if epoch % probe_every == 0 or epoch == 1 or epoch == 24:
            model.eval()
            rec_v, con_v = probe_loss(model, x_val, r.TRAIN['alpha'])
            curve.append(dict(epoch=epoch, train_loss=float(loss), probe_every=probe_every,
                              val_reconstruction=rec_v, val_contrastive=con_v))
            print(json.dumps({'event': 'epoch', 'dataset': name, **curve[-1]}), flush=True)
    out = {'dataset': name, 'tables': tables, 'bits': bits, 'max_epochs': max_epochs,
           'curve': curve, 'code_commit': r.git_head()}
    (r.JOBROOT / 'results' / 'convergence_v1').mkdir(parents=True, exist_ok=True)
    (r.JOBROOT / 'results' / 'convergence_v1' / f'{name}_L{tables}_c{bits}.json').write_text(
        json.dumps(out, indent=2))


if __name__ == '__main__':
    name, tables, bits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    run(name, tables, bits)
