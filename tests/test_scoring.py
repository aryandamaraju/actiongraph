from eval import load_meetings
from scoring import due_key, grounded, ground_filter, owner_key, score_all


def perfect(meetings):
    return {
        m["id"]: [
            {"task": g["task"], "owners": g["owners"], "due": g["due"], "evidence": g["src"]}
            for g in m["gold"]
        ]
        for m in meetings
    }


def test_perfect_predictions_score_one():
    ms = load_meetings("all")
    o = score_all(ms, perfect(ms))["overall"]
    assert o["f1"] == 1.0 and o["strict_f1"] == 1.0
    assert o["owner_acc"] == 1.0 and o["due_acc"] == 1.0 and o["grounded_rate"] == 1.0


def test_empty_predictions_score_zero_recall():
    ms = load_meetings("all")
    r = score_all(ms, {})
    assert r["overall"]["recall"] == 0.0


def test_dropped_item_lowers_recall_only():
    ms = load_meetings("all")
    p = perfect(ms)
    p["m01"] = p["m01"][1:]
    o = score_all(ms, p)["overall"]
    assert o["recall"] < 1.0 and o["precision"] == 1.0


def test_invented_item_lowers_precision_and_is_counted_on_empty_meetings():
    ms = load_meetings("all")
    p = perfect(ms)
    p["m04"] = [{"task": "Build dark mode", "owners": [], "due": None, "evidence": "dark mode in the admin panel"}]
    r = score_all(ms, p)
    assert r["overall"]["precision"] < 1.0
    assert r["items_on_empty_meetings"] == 1


def test_wrong_owner_and_due_are_caught_but_task_still_matches():
    ms = load_meetings("all")
    p = perfect(ms)
    p["m01"][0]["owners"] = ["Dana"]
    p["m01"][1]["due"] = "Monday"
    o = score_all(ms, p)["overall"]
    assert o["recall"] == 1.0
    assert o["owner_acc"] < 1.0 and o["due_acc"] < 1.0 and o["strict_f1"] < 1.0


def test_due_and_owner_normalization():
    assert due_key("by Friday") == due_key("Friday")
    assert due_key("End of the sprint") == due_key("end of sprint")
    assert due_key("Friday") != due_key("next Friday")
    assert due_key(None) is None and due_key("") is None
    assert owner_key(["Marcus Lee"]) == owner_key(["marcus"])
    assert owner_key(["Mika", "Devon"]) == owner_key(["devon", "mika"])
    assert owner_key([]) == owner_key(None)


def test_grounding_and_filter():
    t = "Priya: And I'll send the revised timeline to leadership tomorrow."
    assert grounded("I'll send the revised timeline to leadership tomorrow", t)
    assert grounded("Priya: and I’ll send the revised timeline", t)
    assert not grounded("I will email the board a summary", t)
    keep, dropped = ground_filter(
        [{"evidence": "send the revised timeline"}, {"evidence": "made up quote"}], t)
    assert len(keep) == 1 and dropped == 1


def test_one_prediction_cannot_satisfy_two_gold_items():
    ms = [m for m in load_meetings("all") if m["id"] == "m01"]
    p = {"m01": [{"task": "Fix verification bug and send timeline", "owners": ["Marcus"], "due": "Wednesday", "evidence": "x"}]}
    s = score_all(ms, p)["per_meeting"]["m01"]
    assert s["matched"] == 1
