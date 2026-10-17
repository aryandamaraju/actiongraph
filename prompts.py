"""Prompt versions. v1 is the naive baseline; v2 adds an explicit definition, rules and two
short worked examples. The examples are NOT taken from data/meetings.jsonl."""

OUTPUT_SHAPE = (
    'Return only a JSON object: {"items": [ {"task": str, "owners": [str], "due": str or null, '
    '"evidence": str} ]}.'
)

V1 = (
    "You extract action items from meeting transcripts. "
    + OUTPUT_SHAPE
    + " task is a short description, owners lists the people responsible, due is the deadline if "
    "one was mentioned, and evidence is a quote from the transcript."
)

V2 = (
    """You extract action items from a meeting transcript. Each line starts with the speaker's name.

An action item is a specific thing someone committed to do, or was asked to do and accepted, after the meeting.

Include:
- Commitments like "I'll send it Friday", and requests that the person accepts ("Can you do X?" "Sure.").
- Tasks everyone agrees must happen but nobody owns yet: include them with an empty owners list.
- Conditional commitments ("If legal signs off, I'll add the slide").
- Open items carried over from earlier ("Still owe you that, I'll send it tomorrow").

Leave out:
- Ideas and maybes that nobody committed to ("we should probably", "maybe", "someday", "let's revisit").
- Work already finished ("I already sent it", "that's done").
- Tasks that were cancelled, declined, or put on hold and not picked up by someone else.
- Topics to discuss later, and plain status updates.
- A meeting with no commitments gets {"items": []}.

Fields:
- task: a short imperative phrase, for example "Send the revised timeline to leadership".
- owners: the first names of the people who will do it. "I'll ..." means the speaker. If the task was reassigned or corrected, use the final owner. If several people share it, list them all. If nobody owns it, use [].
- due: the time expression exactly as spoken ("Friday", "next week", "end of day Friday", "the 25th", "noon Friday"). If the speaker corrected a date, use the corrected one. Use null if no time was given. Never convert to a calendar date and never put a condition here.
- evidence: one short quote copied word for word from the transcript where the commitment was made. Do not include the speaker label.

Example 1
Transcript:
Ana: Should we redo the pricing page? Maybe.
Ben: I'll send you the new screenshots Thursday.
Ana: Great. Also Ben, can you book the room?
Ben: Already done.
Output:
{"items": [{"task": "Send the new screenshots to Ana", "owners": ["Ben"], "due": "Thursday", "evidence": "I'll send you the new screenshots Thursday."}]}

Example 2
Transcript:
Cy: Someone has to renew the domain.
Di: Can Eli do it?
Eli: Fine, I'll renew it before the 1st.
Cy: And we're dropping the newsletter redesign.
Output:
{"items": [{"task": "Renew the domain", "owners": ["Eli"], "due": "before the 1st", "evidence": "Fine, I'll renew it before the 1st."}]}

"""
    + OUTPUT_SHAPE
)

V3 = V2.replace("\nFields:\n", '\nAlso:\n- Read every speaker turn. Small commitments count ("I\'ll update the ticket after this call"), and a request counts when the person accepts it, even with a brief "Sure".\n- If something clearly needs doing and nobody takes it ("Someone needs to fix the photo", "Yeah, someone should"), include it with owners [].\n- Never list the same task twice. If a request is handed to a different person, only the new owner gets the task.\n- If different times are mentioned for one task, use the one the owner agreed to last.\n\nFields:\n', 1)

VERSIONS = {"v1": V1, "v2": V2, "v3": V3}


def user_message(transcript):
    return "Transcript:\n\n" + transcript.strip() + "\n\nReturn the JSON now."
