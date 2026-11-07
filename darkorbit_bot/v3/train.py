"""
Train the V3 ObjectTokenPolicy from recorded sessions.

    python -m darkorbit_bot.v3.train                       # all sessions, defaults
    python -m darkorbit_bot.v3.train --epochs 40 --awr     # reward-weighted
    python -m darkorbit_bot.v3.train --init models/v3/policy.pt   # continue training

Validation is split by session (see dataset.split_sessions). The best
validation epoch is saved atomically to --out.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import build_session, apply_awr, split_sessions, ChunkDataset
from .model import ObjectTokenPolicy, PolicyConfig, save_policy, load_policy
from .schema import SESSIONS_DIR, DEFAULT_POLICY_PATH, MAX_OBJECTS
from .session_io import list_sessions

NONE_TARGET_WEIGHT = 0.3   # most frames have no upcoming click


def button_pos_weight(sessions, parts, num_buttons: int, max_weight: float = 1.0) -> torch.Tensor:
    """pos_weight per button (left, right, keys) from train labels, clamped to
    [1, max_weight]. Default 1.0 = plain BCE: probabilities stay calibrated, so
    the runtime 0.5 threshold means "more likely pressed than not". Balancing
    (max_weight > 1) inflates probabilities and makes the bot over-click."""
    pos = np.zeros(num_buttons); tot = 0.0
    for k, idx in parts:
        s = sessions[k]
        m = s.action_mask[idx][..., None]
        pos += (s.actions[idx][:, :, 2:2 + num_buttons] * m).sum(axis=(0, 1))
        tot += float(m.sum())
    neg = tot - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, max(1.0, max_weight)), dtype=torch.float32)


def compute_loss(model, batch, pos_weight, device):
    b = {k: v.to(device) for k, v in batch.items()}
    out = model(b['player'], b['prev_action'], b['objects'], b['mask'])
    w = b['weight']                                  # [B]
    am = b['action_mask']                            # [B, C]
    steps = am.sum(dim=1).clamp(min=1.0)

    # Mouse: L1 in screen space on valid future steps
    mouse = torch.sigmoid(out['mouse_logits'])
    mouse_l = (mouse - b['actions'][:, :, 0:2]).abs().sum(-1)            # [B, C]
    mouse_loss = ((mouse_l * am).sum(1) / steps * w).sum() / w.sum()

    # Buttons + keys: BCE
    nb = out['button_logits'].shape[-1]
    bce = F.binary_cross_entropy_with_logits(
        out['button_logits'], b['actions'][:, :, 2:2 + nb],
        pos_weight=pos_weight.to(device), reduction='none').mean(-1)     # [B, C]
    button_loss = ((bce * am).sum(1) / steps * w).sum() / w.sum()

    # Target pointer
    tgt = b['target']
    ce = F.cross_entropy(out['target_logits'], tgt, reduction='none')
    tw = w * torch.where(tgt == MAX_OBJECTS, NONE_TARGET_WEIGHT, 1.0)
    target_loss = (ce * tw).sum() / tw.sum()

    loss = mouse_loss + button_loss + 0.5 * target_loss

    with torch.no_grad():
        has_click = tgt < MAX_OBJECTS
        pred_t = out['target_logits'].argmax(1)
        metrics = {
            'loss': loss.item(), 'mouse_l1': mouse_loss.item(),
            'buttons': button_loss.item(), 'target': target_loss.item(),
            'target_acc_on_clicks': (pred_t[has_click] == tgt[has_click]).float().mean().item()
            if has_click.any() else float('nan'),
            'lmb_acc': ((torch.sigmoid(out['button_logits'][:, 0, 0]) > 0.5).float()
                        == b['actions'][:, 0, 2]).float().mean().item(),
        }
    return loss, metrics


def run_epoch(model, loader, pos_weight, device, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    agg, n = {}, 0
    for batch in loader:
        with torch.set_grad_enabled(train):
            loss, m = compute_loss(model, batch, pos_weight, device)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        bs = batch['weight'].shape[0]
        for k, v in m.items():
            if not math.isnan(v):
                agg[k] = agg.get(k, 0.0) + v * bs
                agg[k + '_n'] = agg.get(k + '_n', 0) + bs
        n += bs
    return {k: agg[k] / agg[k + '_n'] for k in agg if not k.endswith('_n')}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the V3 single policy")
    ap.add_argument('--sessions-dir', default=str(SESSIONS_DIR))
    ap.add_argument('--out', default=str(DEFAULT_POLICY_PATH))
    ap.add_argument('--init', default=None, help='continue from this V3 checkpoint')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--awr', action='store_true', help='reward-weighted imitation (HUD/combat-log rewards)')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--patience', type=int, default=8, help='stop after N epochs without val improvement')
    ap.add_argument('--button-pos-weight-max', type=float, default=1.0,
                    help='>1 balances rare buttons/keys (miscalibrates; raise the runtime threshold if used)')
    args = ap.parse_args(argv)

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    paths = list_sessions(Path(args.sessions_dir))
    if not paths:
        raise SystemExit(f"No sessions in {args.sessions_dir}. Record some play first "
                         f"(python -m darkorbit_bot.v3.agent --mode record).")

    print(f"[V3-TRAIN] Building {len(paths)} session(s)...")
    sessions = []
    for p in paths:
        s = build_session(p)
        if args.awr:
            apply_awr(s)
        print(f"   {s.session_id}: {len(s.t)} frames, {int(s.sample_ok.sum())} samples, "
              f"{int((s.target < MAX_OBJECTS).sum())} with a click target")
        sessions.append(s)

    train_parts, val_parts, split_info = split_sessions(sessions, args.val_frac)
    train_ds, val_ds = ChunkDataset(sessions, train_parts), ChunkDataset(sessions, val_parts)
    print(f"[V3-TRAIN] Split {split_info['mode']}: train {len(train_ds)} / val {len(val_ds)} samples")
    if len(train_ds) == 0:
        raise SystemExit("No training samples (need frames where you played).")

    model = load_policy(args.init, device) if args.init else ObjectTokenPolicy(PolicyConfig()).to(device)
    pos_weight = button_pos_weight(sessions, train_parts, model.cfg.num_buttons, args.button_pos_weight_max)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size) if len(val_ds) else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(train_dl))
    warmup = min(200, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    best, best_state, best_epoch, bad = float('inf'), None, -1, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, pos_weight, device, opt, sched)
        va = run_epoch(model, val_dl, pos_weight, device) if val_dl else tr
        history.append({'epoch': epoch, 'train': tr, 'val': va})
        print(f"[V3-TRAIN] epoch {epoch:3d}  train {tr['loss']:.4f}  val {va['loss']:.4f}  "
              f"mouse {va['mouse_l1']:.3f}  lmb_acc {va.get('lmb_acc', float('nan')):.2f}  "
              f"target_acc {va.get('target_acc_on_clicks', float('nan')):.2f}  ({time.time() - t0:.1f}s)")
        if va['loss'] < best - 1e-5:
            best, best_epoch, bad = va['loss'], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[V3-TRAIN] No val improvement for {args.patience} epochs - stopping")
                break

    model.load_state_dict(best_state)
    save_policy(model, args.out, extra={
        'trained': time.strftime('%Y-%m-%d %H:%M:%S'), 'best_epoch': best_epoch, 'best_val_loss': best,
        'sessions': [s.session_id for s in sessions], 'split': split_info, 'awr': args.awr,
        'history': history})
    print(f"[V3-TRAIN] Saved best epoch {best_epoch} (val {best:.4f}) to {args.out}")


if __name__ == '__main__':
    main()
