# ActionGraph

Paste a meeting transcript, get back a checked list of action items: the task, who owns it, when it is due, and the exact quote from the transcript that proves it.

**Live demo:** https://actiongraph.vercel.app (free tier; paste a transcript or use one of the three examples)

This is a small, honest evaluation project. The model call is the easy part. The work is in defining what counts as an action item, labeling 30 meetings by that definition, scoring the output with a script, and showing what a better prompt and a bigger model change.

## Results

Scored on **20 held-out meetings (74 action items)** that no prompt was tuned on. Each setup was run once at temperature 0.

| setup | tasks found right (F1) | owner right | due right | task + owner + due all right (F1) | quotes found in transcript |
|---|---|---|---|---|---|
| v1: basic prompt, gpt-oss-20b | 93% | 100% | 67% | **62%** | 88% |
| v2: rules + 2 examples, gpt-oss-20b | 96% | 97% | 90% | **85%** | 100% |
| v3: v2 + 4 extra rules, gpt-oss-20b | 97% | 97% | 93% | **88%** | 100% |
| v3, gpt-oss-120b | 98% | 100% | 97% | **95%** | 100% |

How to read it: "owner right" and "due right" are measured only on tasks the model found. The last two columns are the strict ones: a task only counts as fully right when the task, owner and due date all match the labels.

- **The basic prompt finds almost every task but gets dates wrong.** It converts "Friday" into a calendar date, often with the wrong year (for example `2024-10-06`), so only 67% of its due dates match what was said. The improved prompts tell the model to quote the time expression as spoken, and due accuracy goes from 67% to 90-97%.
- **A bigger model helps on the hard cases.** Moving from the 20B to the 120B model takes the strict score from 88% to 95% with the same prompt.
- **Nothing was invented.** On the two meetings with no action items, no setup produced a made-up item.
- **Tuning scores are higher, as expected.** On the 10 meetings used to write v3 (30 items) the strict F1 was v1 66%, v2 90%, v3 97%. That number is inflated because v3 was written after reading those meetings' errors. The table above is the number to trust.

The live demo uses the v3 prompt with gpt-oss-120b.

### What the grounding filter does

Every item must include a quote copied from the transcript, and the app drops any item whose quote cannot be found word for word. That guarantees every quote the user sees is real. As a way to improve accuracy it did not pay off: on the basic prompt it raised "quotes found" from 88% to 100% but also removed correct items (recall 93% to 81%, strict F1 62% to 60%), and for v2 and v3 it changed nothing because their quotes were already exact.

### Where it still fails

Looking at the v3 / 20B errors on the held-out meetings:

- **Small commitments buried in a line are missed.** "I'll tell their office manager to expect it" was missed by every setup, including the 120B model. "I'll write the CEO letter over the weekend" and "if I don't hear back by tomorrow I'll escalate to their controller" were also dropped.
- **Duplicates.** When one person says "somebody should check X" and another later says "I'll check it this afternoon", the model sometimes lists two items, one with no owner.
- **"Now" and "right away" are not treated as dates.** The labels count them as a due time; the models usually answer null. This is partly a labeling choice and I would revisit it.
- **A date attached to the wrong step.** In one meeting the models put a date on the first email instead of on the escalation that follows it.

## How it works

1. `extract.py` sends the transcript to an OpenAI-compatible chat API (Groq by default) in JSON mode with a fixed output shape.
2. The reply is parsed and validated with Pydantic (`schema.py`). If it is not valid JSON or fails validation, the error is sent back to the model for a corrected reply, up to three tries. In all 110 meeting runs behind this README it was never needed.
3. Optionally, items whose evidence quote is not found in the transcript are dropped.
4. `app.py` serves this as a small FastAPI app with a one-page front end (`static/index.html`), deployed on Vercel.

## What counts as an action item (the labeling policy)

An action item is a specific thing someone committed to do, or was asked to do and accepted, after the meeting.

- **Included:** commitments ("I'll send it Friday"), accepted requests ("Can you do X?" "Sure."), tasks everyone agrees must happen but nobody owns yet (owner left empty), conditional commitments ("if legal signs off, I'll add the slide"), and carried-over items.
- **Excluded:** ideas and maybes nobody committed to, work already finished, tasks that were cancelled or declined and not picked up, topics to discuss later, and plain status updates.
- **Owner:** the person who will do it; the final owner if it was reassigned; several people if shared; empty if nobody.
- **Due:** the time expression as spoken ("Friday", "the 25th", "noon Friday"); the corrected one if the speaker corrected themselves; empty if no time was given.

## The data

`data/meetings.jsonl` has 30 synthetic meeting transcripts (about 120 words each) with 104 labeled action items. The transcripts were drafted with an AI assistant and labeled to the policy above; no real meetings or people are involved. Each meeting has tags for the tricky situation it tests (cancelled task, reassigned task, vague owner, hedged "maybe", conditional, noisy, correction, no action items, and so on).

- **10 `dev` meetings** (30 items) were used to write and tune the prompts.
- **20 `test` meetings** (74 items) were run once per setup after the prompts were fixed.

Every label carries a quote from its transcript, and a test checks that each quote, owner name and due phrase really appears in the transcript.

## How scoring works

`scoring.py` (standard library only):

1. **Match.** A predicted item counts as the right task for a labeled item when its task text contains that item's key terms (a short hand-written list of alternatives per label). When several fit, the closest wording wins, and each item is used once.
2. **Fields.** For matched items, owners are compared as a set of first names and due dates as a set of words after dropping filler like "by" and "the". Both are strict.
3. **Quotes.** Separately, each item's evidence quote must appear word for word in the transcript, ignoring case and punctuation.

This is deliberately simple and reproducible, and it has costs: it is lenient about wording (it cannot tell a subtly wrong paraphrase from a right one) and the key-term lists are hand-written.

## What changed after I saw the test results

One thing. A correct prediction ("Fix ticket 4412 and push to staging") was scored as a miss because the label's key terms did not include the ticket number. I added the ticket numbers as alternatives for that meeting's labels and re-scored every saved run with the corrected labels. Only one number changed (v3 / 20B strict F1, 87% to 88%). Prompts were not changed after the test run.

## Honest limits

- Small test set: 74 items, so one item is about 1.4 points, and the differences between v2 and v3 are within noise. The gaps between v1 and the others, and between the 20B and 120B models, are larger.
- Single run per setup. Temperature is 0, but the API is not guaranteed to be perfectly repeatable.
- Short, clean, synthetic transcripts. Real meetings are longer, messier, and have speaker-attribution problems that this set barely touches.
- The labels were written by one person (with an assistant) and some are judgment calls; the policy above is the tie-breaker.

## Run it yourself

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows
pip install -r requirements-dev.txt
copy .env.example .env                                 # then put your own Groq key in .env
python -m pytest -q                                    # no API key needed
python eval.py --split dev  --prompt v3 --show-errors  # tuning meetings
python eval.py --split test --prompt v3                # held-out meetings
python eval.py --rescore results/test_v3_openai-gpt-oss-120b.json --markdown   # re-score saved output, no key
uvicorn app:app --reload                               # run the demo locally
```

`pytest` runs the data checks, the scorer tests, the extractor tests (with a mocked API), and re-scores the saved model outputs in `results/` against floors in `results/floors.json`. GitHub Actions runs the same thing on every push, with no API key.

## Deployment

The demo runs on Vercel's free tier with the Groq free API and `openai/gpt-oss-120b` using the v3 prompt. Settings are environment variables (`LLM_API_KEY`, `LLM_MODEL`, `PROMPT_VERSION`); the key is never in the repo. The endpoint caps transcripts at 12,000 characters and limits each address to 10 requests per 10 minutes to protect the free quota (best effort, per serverless instance). Transcripts are sent to the model API and are not stored by this app. A first request after a quiet period can be slower while the function starts.

Groq retires models regularly (`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` were shut down on 2026-08-16), so the model is a setting, not hard-coded, and this project uses the currently listed `gpt-oss` models.

## Layout

```
app.py            FastAPI app (POST /extract, GET /health)
extract.py        model call, JSON repair loop, grounding filter
prompts.py        prompt versions v1, v2, v3
schema.py         output validation
scoring.py        matching and metrics
eval.py           run or re-score the evaluation
data/meetings.jsonl   30 labeled meetings
results/          saved model outputs for every run in this README
tests/            pytest suite
static/index.html the one-page demo
```
