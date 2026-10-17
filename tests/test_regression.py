"""Re-score the saved model outputs in results/ (no API key needed) and make sure the
scores have not fallen below the floors in results/floors.json."""
import json
from pathlib import Path

from eval import items_for_scoring, load_meetings
from scoring import score_all

RESULTS = Path(__file__).resolve().parent.parent / "results"


def test_saved_results_meet_their_floors():
    floors_path = RESULTS / "floors.json"
    assert floors_path.exists(), 'results/floors.json is missing, so saved results are not being checked'
    floors = json.loads(floors_path.read_text(encoding="utf-8"))
    for name, spec in floors.items():
        saved = json.loads((RESULTS / spec["file"]).read_text(encoding="utf-8"))
        meetings = load_meetings(saved["split"])
        res = score_all(meetings, items_for_scoring(saved, meetings, spec.get("ground_filter", False)))
        for metric, floor in spec["min"].items():
            assert res["overall"][metric] >= floor, (name, metric, res["overall"][metric], floor)
