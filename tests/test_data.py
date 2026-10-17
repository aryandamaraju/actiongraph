from eval import load_meetings

TAGS = {
    "simple", "self_assign", "vague_owner", "non_commitment", "cancelled", "reassigned",
    "completed", "conditional", "multi_owner", "relative_due", "noisy", "correction",
    "no_items", "long", "small_talk", "confirm_later", "carryover", "external_people",
}


def test_sizes():
    ms = load_meetings("all")
    assert len(ms) == 30
    assert len(load_meetings("dev")) == 10
    assert len(load_meetings("test")) == 20
    assert len({m["id"] for m in ms}) == 30


def test_gold_is_grounded_in_transcripts():
    for m in load_meetings("all"):
        t = m["transcript"]
        assert set(m["tags"]) <= TAGS, m["id"]
        assert ("no_items" in m["tags"]) == (len(m["gold"]) == 0), m["id"]
        for g in m["gold"]:
            assert g["src"] in t, (m["id"], g["src"])
            assert g["due"] is None or g["due"].lower() in t.lower(), (m["id"], g["due"])
            assert g["kw"], (m["id"], g["task"])
            for o in g["owners"]:
                assert o in t, (m["id"], o)
