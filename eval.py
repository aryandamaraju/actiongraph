"""Run or re-score the ActionGraph evaluation.

Run the model:   python eval.py --split dev  --prompt v1
                 python eval.py --split test --prompt v2 --model openai/gpt-oss-120b
Re-score saved:  python eval.py --rescore results/test_v2_openai-gpt-oss-20b.json --ground-filter
No API key is needed to re-score; this is what CI does.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scoring import ground_filter, score_all

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_meetings(split="all"):
    lines = (ROOT / "data" / "meetings.jsonl").read_text(encoding="utf-8").splitlines()
    ms = [json.loads(line) for line in lines if line.strip()]
    return ms if split == "all" else [m for m in ms if m["split"] == split]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def run_model(meetings, version, model, delay):
    import extract

    preds, errors = {}, {}
    for i, m in enumerate(meetings, 1):
        try:
            preds[m["id"]] = extract.extract(m["transcript"], version=version, model=model)
        except Exception as e:  # keep going; a failed meeting counts as zero items
            errors[m["id"]] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  {m['id']}  {i}/{len(meetings)}", flush=True)
        time.sleep(delay)
    return preds, errors


def items_for_scoring(saved, meetings, use_filter):
    out = {}
    for m in meetings:
        r = saved["predictions"].get(m["id"])
        if not r:
            continue
        items = r["items"]
        if use_filter:
            items, _ = ground_filter(items, m["transcript"])
        out[m["id"]] = items
    return out


def pct(x):
    return f"{100 * x:.0f}%"


def print_report(saved, meetings, use_filter, show_errors):
    res = score_all(meetings, items_for_scoring(saved, meetings, use_filter))
    o = res["overall"]
    label = f"split={saved['split']} prompt={saved['version']} model={saved['model']} ground_filter={use_filter}"
    print("\n" + label)
    print(f"{o['meetings']} meetings, {o['gold_items']} gold items, {o['pred_items']} predicted")
    print(f"  task precision {pct(o['precision'])}   recall {pct(o['recall'])}   F1 {pct(o['f1'])}")
    print(f"  owner correct {pct(o['owner_acc'])}   due correct {pct(o['due_acc'])}   (of matched tasks)")
    print(f"  strict precision {pct(o['strict_precision'])}   recall {pct(o['strict_recall'])}   F1 {pct(o['strict_f1'])}")
    print(f"  evidence quotes found in transcript: {pct(o['grounded_rate'])}")
    print(f"  items invented on {res['empty_meetings']} meetings with no action items: {res['items_on_empty_meetings']}")
    if saved.get("errors"):
        print(f"  FAILED meetings (counted as zero items): {sorted(saved['errors'])}")
    attempts = [r["meta"]["attempts"] for r in saved["predictions"].values()]
    if attempts:
        retried = sum(1 for a in attempts if a > 1)
        print(f"  needed a JSON repair retry: {retried}/{len(attempts)}")
    print("  by tag (F1, meetings):", ", ".join(
        f"{t} {pct(v['f1'])}/{v['meetings']}" for t, v in res["by_tag"].items()))
    if show_errors:
        for m in meetings:
            s = res["per_meeting"][m["id"]]
            if s["missed"] or s["extra"] or s["owner_ok"] < s["matched"] or s["due_ok"] < s["matched"]:
                print(f"\n  [{m['id']}] tags={m['tags']}")
                for t in s["missed"]:
                    print(f"    MISSED: {t}")
                for it in s["extra"]:
                    print(f"    EXTRA : {it['task']} | {it['owners']} | {it['due']}")
                for p in s["pairs"]:
                    g = m["gold"][p["gold"]]
                    if not (p["owner_ok"] and p["due_ok"]):
                        # pred index refers to the list passed to scoring (possibly filtered)
                        items = items_for_scoring(saved, [m], use_filter)[m["id"]]
                        q = items[p["pred"]]
                        print(f"    FIELDS: {g['task']}: gold {g['owners']}/{g['due']} vs pred {q['owners']}/{q['due']}")
    return res


def markdown_row(saved, res, use_filter):
    o = res["overall"]
    name = f"{saved['version']}{' + grounding filter' if use_filter else ''} ({saved['model'].split('/')[-1]})"
    return (f"| {name} | {pct(o['precision'])} | {pct(o['recall'])} | {pct(o['f1'])} | "
            f"{pct(o['owner_acc'])} | {pct(o['due_acc'])} | {pct(o['strict_f1'])} | {pct(o['grounded_rate'])} |")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a narrow encoding
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--prompt", choices=["v1", "v2", "v3"], default="v2")
    ap.add_argument("--model", default=None)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between API calls")
    ap.add_argument("--rescore", nargs="+", help="re-score saved result files without calling the model")
    ap.add_argument("--ground-filter", action="store_true", help="drop items whose evidence is not in the transcript")
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="also print a markdown table row")
    args = ap.parse_args()

    if args.rescore:
        rows = []
        for path in args.rescore:
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
            meetings = load_meetings(saved["split"])
            res = print_report(saved, meetings, args.ground_filter, args.show_errors)
            rows.append(markdown_row(saved, res, args.ground_filter))
        if args.markdown:
            print("\n| setup | precision | recall | F1 | owner | due | strict F1 | quotes found |")
            print("|---|---|---|---|---|---|---|---|")
            print("\n".join(rows))
        return

    import os

    model = args.model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    meetings = load_meetings(args.split)
    print(f"Running {args.prompt} on {len(meetings)} {args.split} meetings with {model} ...")
    preds, errors = run_model(meetings, args.prompt, model, args.delay)
    saved = {
        "split": args.split,
        "version": args.prompt,
        "model": model,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predictions": preds,
        "errors": errors,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.split}_{args.prompt}_{slug(model)}.json"
    out.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", out.relative_to(ROOT))
    res = print_report(saved, meetings, args.ground_filter, args.show_errors)
    if args.markdown:
        print(markdown_row(saved, res, args.ground_filter))
    if errors and len(errors) == len(meetings):
        sys.exit("Every meeting failed. Check LLM_API_KEY / LLM_MODEL in .env.")


if __name__ == "__main__":
    main()
