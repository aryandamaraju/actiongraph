import json

import extract

VALID = json.dumps({"items": [{"task": "Send the report", "owners": "Sam", "due": "null", "evidence": "I'll send it"}]})


class Fake:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, body, cfg, deadline=None):
        self.calls.append(body)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"choices": [{"message": {"content": reply}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def run(replies, **kw):
    fake = Fake(replies)
    old_post, old_key = extract._post, extract.os.environ.get("LLM_API_KEY")
    extract._post = fake
    extract.os.environ["LLM_API_KEY"] = "test-key"
    try:
        return extract.extract("Sam: I'll send it", **kw), fake
    finally:
        extract._post = old_post
        if old_key is None:
            extract.os.environ.pop("LLM_API_KEY", None)
        else:
            extract.os.environ["LLM_API_KEY"] = old_key


def test_valid_reply_is_normalized():
    out, fake = run([VALID])
    it = out["items"][0]
    assert it["owners"] == ["Sam"] and it["due"] is None
    assert out["meta"]["attempts"] == 1


def test_bad_json_triggers_repair_retry():
    out, fake = run(["not json at all", "```json\n" + VALID + "\n```"])
    assert out["meta"]["attempts"] == 2
    assert "not valid" in fake.calls[1]["messages"][-1]["content"]


def test_schema_error_triggers_retry():
    bad = json.dumps({"items": [{"task": "x", "evidence": ""}]})
    out, fake = run([bad, VALID])
    assert out["meta"]["attempts"] == 2


def test_gives_up_after_three_tries():
    try:
        run(["nope", "nope", "nope"])
    except extract.LLMError as e:
        assert "3 tries" in str(e)
    else:
        raise AssertionError("expected LLMError")


def test_json_mode_fallback_on_400():
    err = extract.LLMError('HTTP 400: {"error":{"message":"response_format json_object is not supported"}}')
    out, fake = run([err, VALID])
    assert "response_format" in fake.calls[0] and "response_format" not in fake.calls[1]


def test_ground_filter_drops_made_up_evidence():
    reply = json.dumps({"items": [
        {"task": "Send the report", "owners": ["Sam"], "evidence": "I'll send it"},
        {"task": "Book a venue", "owners": ["Sam"], "evidence": "we should book a venue"},
    ]})
    out, _ = run([reply], ground=True)
    assert len(out["items"]) == 1 and out["meta"]["dropped_ungrounded"] == 1
