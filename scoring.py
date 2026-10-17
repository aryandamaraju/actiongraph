"""Scoring for ActionGraph. Standard library only, so it runs anywhere (including CI).

How a prediction is scored against the gold labels of one meeting:

1. MATCH. A predicted item counts as "the right task" for a gold item when its
   task text contains the gold item's key terms (gold["kw"]: a list of groups, and
   every group needs at least one of its alternatives to appear at a word start).
   When several predictions fit, the closest wording wins; each gold item and each
   prediction is used at most once.
2. FIELDS. For matched pairs, owners and due are compared strictly:
   owners as a set of lowercase first names, due as a set of words after dropping
   filler like "by", "on", "the", "this".
3. GROUNDING. Separately, each prediction's evidence quote must appear word for word
   in the transcript (ignoring case and punctuation).
"""
import re
from collections import defaultdict

DUE_STOP = {"by", "on", "before", "until", "the", "this", "at", "in"}
WORD_STOP = {"the", "a", "an", "to", "of", "and", "for", "on", "in", "with", "it", "that", "is", "be"}


def norm_text(s):
    s = (s or "").lower().replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def due_key(s):
    if s is None or not str(s).strip():
        return None
    toks = frozenset(t for t in norm_text(str(s)).split() if t not in DUE_STOP)
    return toks or None


def owner_key(owners):
    keys = set()
    for o in owners or []:
        n = norm_text(str(o))
        if not n:
            continue
        parts = n.split()
        keys.add(n if parts[0] in ("speaker", "unknown") else parts[0])
    return frozenset(keys)


def _tokens(s):
    return {t for t in norm_text(s).split() if t not in WORD_STOP}


def _jaccard(a, b):
    a, b = _tokens(a), _tokens(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def kw_ok(task, kw):
    text = " " + norm_text(task)
    for group in kw:
        if not any(re.search(r"(?<![a-z0-9])" + re.escape(norm_text(alt)), text) for alt in group):
            return False
    return True


def grounded(evidence, transcript):
    ev = norm_text(evidence)
    return bool(ev) and ev in norm_text(transcript)


def match_items(gold, pred):
    """Return list of (gold_index, pred_index) pairs, one-to-one."""
    cands = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if kw_ok(p.get("task", ""), g["kw"]):
                cands.append((_jaccard(g["task"], p.get("task", "")), gi, pi))
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_g, used_p, pairs = set(), set(), []
    for _, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi))
    return sorted(pairs)


def score_meeting(meeting, pred_items):
    gold = meeting["gold"]
    pairs = match_items(gold, pred_items)
    owner_ok = due_ok = strict = 0
    detail = []
    for gi, pi in pairs:
        g, p = gold[gi], pred_items[pi]
        o = owner_key(p.get("owners")) == owner_key(g["owners"])
        d = due_key(p.get("due")) == due_key(g["due"])
        owner_ok += o
        due_ok += d
        strict += o and d
        detail.append({"gold": gi, "pred": pi, "owner_ok": o, "due_ok": d})
    n_ground = sum(grounded(p.get("evidence", ""), meeting["transcript"]) for p in pred_items)
    matched_g = {gi for gi, _ in pairs}
    matched_p = {pi for _, pi in pairs}
    return {
        "gold": len(gold),
        "pred": len(pred_items),
        "matched": len(pairs),
        "owner_ok": owner_ok,
        "due_ok": due_ok,
        "strict": strict,
        "grounded": n_ground,
        "pairs": detail,
        "missed": [gold[i]["task"] for i in range(len(gold)) if i not in matched_g],
        "extra": [pred_items[i] for i in range(len(pred_items)) if i not in matched_p],
    }


def _prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else (1.0 if n_gold == 0 else 0.0)
    r = tp / n_gold if n_gold else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def aggregate(per_meeting):
    t = defaultdict(int)
    for s in per_meeting:
        for k in ("gold", "pred", "matched", "owner_ok", "due_ok", "strict", "grounded"):
            t[k] += s[k]
    p, r, f = _prf(t["matched"], t["pred"], t["gold"])
    sp, sr, sf = _prf(t["strict"], t["pred"], t["gold"])
    return {
        "meetings": len(per_meeting),
        "gold_items": t["gold"],
        "pred_items": t["pred"],
        "precision": p,
        "recall": r,
        "f1": f,
        "owner_acc": t["owner_ok"] / t["matched"] if t["matched"] else 0.0,
        "due_acc": t["due_ok"] / t["matched"] if t["matched"] else 0.0,
        "strict_precision": sp,
        "strict_recall": sr,
        "strict_f1": sf,
        "grounded_rate": t["grounded"] / t["pred"] if t["pred"] else 1.0,
    }


def score_all(meetings, predictions):
    """predictions: {meeting_id: [item dicts]} . Meetings without a prediction count as empty."""
    per = {}
    for m in meetings:
        per[m["id"]] = score_meeting(m, predictions.get(m["id"], []))
    overall = aggregate(list(per.values()))
    by_tag = {}
    tags = sorted({t for m in meetings for t in m["tags"]})
    for tag in tags:
        sel = [per[m["id"]] for m in meetings if tag in m["tags"]]
        by_tag[tag] = aggregate(sel)
    empties = [m for m in meetings if not m["gold"]]
    empty_fp = sum(per[m["id"]]["pred"] for m in empties)
    return {"overall": overall, "by_tag": by_tag, "per_meeting": per,
            "empty_meetings": len(empties), "items_on_empty_meetings": empty_fp}


def ground_filter(items, transcript):
    """Drop items whose evidence quote is not found word for word in the transcript."""
    keep = [it for it in items if grounded(it.get("evidence", ""), transcript)]
    return keep, len(items) - len(keep)
