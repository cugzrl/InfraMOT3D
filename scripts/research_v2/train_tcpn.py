"""训练阶段 B 候选质量网络，并在训练侧留出序列上评估

所有模型选择只看 --holdout 序列，正式 val 只在 infer 阶段使用
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from inframot3d.tcpn.dataset import FrameDataset, collate, to_device
from inframot3d.tcpn.det_metrics import PRAccumulator, ece
from inframot3d.tcpn.matching import iou3d_matrix
from inframot3d.tcpn.model import TCPN, apply_box, loss_fn
from inframot3d.tcpn.sampling import PATTERNS, LaneSampler

VARIANTS = {
    "cand": dict(use_hist=False, use_bev=False, use_relation=False),
    "geom": dict(use_hist=False, use_bev=False),
    "hist": dict(use_hist=True, use_bev=False),
    "bev": dict(use_hist=False, use_bev=True),
    "hist_bev": dict(use_hist=True, use_bev=True, use_site=True),
    "hist_bev_nosite": dict(use_hist=True, use_bev=True, use_site=False),
}


def auc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pos = labels == 1
    if pos.sum() == 0 or (~pos).sum() == 0:
        return None
    from scipy.stats import rankdata

    r = rankdata(scores)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=8, dump=None):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)
    acc_raw, acc_new = PRAccumulator(), PRAccumulator()
    low_raw, low_new = PRAccumulator(), PRAccumulator()
    probs, labels, raws = [], [], []
    exist_p, exist_y, exist_state = [], [], []
    iou_before, iou_after = [], []
    pos_err_before, pos_err_after = [], []
    rows = []
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        prob = torch.sigmoid(out["quality"]).cpu().numpy()
        refined = apply_box(batch["cand_box"], out["box"]).cpu().numpy()
        for b, index in enumerate(batch["index"].tolist()):
            sample = dataset.samples[index]
            n = len(sample["cand_box"])
            p = prob[b, :n]
            raw = sample["cand_score"]
            acc_raw.add(raw, sample["cand_box"], sample["gt_boxes"])
            acc_new.add(p, sample["cand_box"], sample["gt_boxes"])
            low = raw < 0.3
            low_raw.add(raw[low], sample["cand_box"][low], sample["gt_boxes"])
            low_new.add(p[low], sample["cand_box"][low], sample["gt_boxes"])
            q = sample["cand_quality"]
            valid = q >= 0
            probs.append(p[valid])
            labels.append(q[valid])
            raws.append(raw[valid])
            tp = np.where(sample["cand_status"] == 0)[0]
            if len(tp):
                before = np.diag(iou3d_matrix(sample["cand_box"][tp], sample["cand_gt_box"][tp]))
                after = np.diag(iou3d_matrix(refined[b, tp], sample["cand_gt_box"][tp]))
                iou_before.append(before)
                iou_after.append(after)
            if dump is not None:
                rows.append({"sequence_id": sample["sequence_id"], "frame_id": sample["frame_id"], "prob": p.tolist(), "raw": raw.tolist(), "quality": q.tolist(), "status": sample["cand_status"].tolist(), "input_index": sample["cand_input_index"].tolist(), "refined": refined[b, :n].tolist()})
            if "exist" in out:
                m = int(batch["trk_mask"][b].sum())
                e = batch["trk_exist"][b, :m].cpu().numpy()
                ep = torch.sigmoid(out["exist"][b, :m]).cpu().numpy()
                st = batch["trk_state"][b, :m].cpu().numpy()
                exist_p.append(ep)
                exist_y.append(e)
                exist_state.append(st)
                pv = batch["trk_pos_valid"][b, :m].cpu().numpy() > 0
                if pv.any():
                    tgt = batch["trk_pos_target"][b, :m].cpu().numpy()[pv]
                    xy = batch["trk_xy"][b, :m].cpu().numpy()[pv]
                    corr = out["trk_pos"][b, :m].cpu().numpy()[pv]
                    pos_err_before.append(np.linalg.norm(tgt - xy, axis=1))
                    pos_err_after.append(np.linalg.norm(tgt - xy - corr, axis=1))
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    raws = np.concatenate(raws)
    result = {
        "raw": acc_raw.summary(),
        "model": acc_new.summary(),
        "low_raw": low_raw.summary(),
        "low_model": low_new.summary(),
        "quality_auc": auc(labels, probs),
        "raw_auc": auc(labels, raws),
        "quality_ece": ece(probs, labels),
        "raw_ece": ece(raws, labels),
    }
    if iou_before:
        b_, a_ = np.concatenate(iou_before), np.concatenate(iou_after)
        result["box"] = {"iou_before": float(b_.mean()), "iou_after": float(a_.mean()), "frac_improved": float((a_ > b_ + 1e-3).mean()), "frac_ge_0.7_before": float((b_ >= 0.7).mean()), "frac_ge_0.7_after": float((a_ >= 0.7).mean())}
    if exist_p:
        ep, ey, es = np.concatenate(exist_p), np.concatenate(exist_y), np.concatenate(exist_state)
        v = ey >= 0
        result["exist_auc"] = auc(ey[v], ep[v])
        result["exist_by_state"] = {str(s): float(ep[es == s].mean()) for s in np.unique(es)}
        if pos_err_before:
            pb, pa = np.concatenate(pos_err_before), np.concatenate(pos_err_after)
            result["track_pos"] = {"before_median": float(np.median(pb)), "after_median": float(np.median(pa)), "before_p90": float(np.percentile(pb, 90)), "after_p90": float(np.percentile(pa, 90))}
    if dump is not None:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        with open(dump, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--bev", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--query", default="grae", choices=["grae", "gt"])
    parser.add_argument("--eval-query", nargs="+", default=["grae"])
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--bev-pattern", default=None, choices=[None, *PATTERNS])
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    spec = VARIANTS[args.variant]
    need_bev = spec.get("use_bev", False)
    pattern = args.bev_pattern if need_bev else None
    sampler = LaneSampler(ROOT) if pattern in ("lane", "laneline") else None
    train_set = FrameDataset(ROOT / args.samples, ROOT / args.bev, args.train, query=args.query, perturb=args.perturb, seed=args.seed, need_bev=need_bev, pattern=pattern, sampler=sampler)
    if args.max_frames:
        train_set.samples = train_set.samples[: args.max_frames]
    holdouts = {q: FrameDataset(ROOT / args.samples, ROOT / args.bev, args.holdout, query=q, need_bev=need_bev, pattern=pattern, sampler=sampler) for q in args.eval_query}
    model = TCPN(hidden=args.hidden, **spec).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(train_set) + args.batch - 1) // args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.1)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    history = []
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        train_set.epoch = epoch
        model.train()
        loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate, generator=generator, persistent_workers=False)
        logs = []
        start = time.time()
        for batch in loader:
            batch = to_device(batch, device)
            result = model(batch)
            loss, log = loss_fn(result, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            logs.append(log)
        row = {"epoch": epoch, "train": {k: float(np.mean([l[k] for l in logs if k in l])) for k in logs[0]}, "seconds": time.time() - start}
        for q, ds in holdouts.items():
            dump = out / ("holdout_%s_epoch%d.jsonl" % (q, epoch)) if epoch == args.epochs else None
            row["holdout_" + q] = evaluate(model, ds, device, dump=dump)
        history.append(row)
        h = row["holdout_" + args.eval_query[0]]
        print(
            "epoch %d loss %.4f | AP raw %.4f model %.4f | R@0.1FP raw %.4f model %.4f | lowAP raw %.4f model %.4f | qAUC %.4f exist %s"
            % (epoch, row["train"]["total"], h["raw"]["AP"], h["model"]["AP"], h["raw"]["recall@fp0.10"], h["model"]["recall@fp0.10"], h["low_raw"]["AP"], h["low_model"]["AP"], h["quality_auc"] or 0, h.get("exist_auc")),
            flush=True,
        )
        torch.save({"model": model.state_dict(), "variant": args.variant, "spec": spec, "hidden": args.hidden, "epoch": epoch, "args": vars(args)}, out / "last.pth")
        (out / "history.json").write_text(json.dumps({"params": params, "history": history}, indent=2))


if __name__ == "__main__":
    main()
