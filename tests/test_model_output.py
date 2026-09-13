"""Test free OpenRouter models on perception classification."""
import json, urllib.request, os, re, time

KEY = ""
with open(os.path.expanduser("~/.hermes/.env")) as f:
    for line in f:
        if line.startswith("OPENROUTER_API_KEY="):
            KEY = line.strip().split("=", 1)[1]
            break

MODEL = "liquid/lfm-2.5-2.6b:free"
SYSTEM = "You are a JSON classifier. Return ONLY a valid JSON object. No markdown fences, no explanation. Just raw JSON."

TEST_INPUTS = [
    ("The user asked me to remember their birthday is March 15th.", "episodic"),
    ("Deploy the new build to staging at 3 PM.", "procedural"),
    ("The ambient temperature in the server room is 22C.", "perceptual"),
    ("The user seems frustrated with the repeated errors.", "emotional"),
    ("Meeting with Dr. Smith about the dissertation at 2 PM tomorrow.", "episodic"),
    ("Random noise: asdf jkl qwerty 12345", "noise"),
    ("The paper by Vaswani et al. introduced transformers in 2017.", "semantic"),
    ("User preference: always use dark mode, never use emojis.", "semantic"),
    ("The git push failed because the branch was behind origin.", "procedural"),
    ("I feel like this conversation is going nowhere.", "emotional"),
    ("The PDF extraction missed table 3 on page 7.", "procedural"),
    ("Remember to update the ZenBrain schema before the demo.", "procedural"),
]

VALID_CATS = {"episodic", "semantic", "procedural", "emotional", "perceptual", "social", "noise"}
VALID_ACTS = {"persist", "discard", "escalate", "defer"}

print(f"Testing {MODEL} on {len(TEST_INPUTS)} inputs\n")
hdr = f"{'#':>2}  {'Status':<6} {'Cat':<14} {'Exp':<14} {'Sal':>5} {'Action':<10} {'Lat':>5}"
print(hdr)
print("-" * len(hdr))

ok = 0
for i, (stim, expected) in enumerate(TEST_INPUTS):
    prompt = (
        f'Classify: "{stim}"\n\n'
        'Return JSON: {"category": "episodic|semantic|procedural|emotional|perceptual|social|noise", '
        '"salience": 0.0-1.0, "action": "persist|discard|escalate|defer", "reasoning": "one sentence"}'
    )

    payload = json.dumps({
        "model": MODEL,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + KEY,
        },
    )

    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        raw = data["choices"][0]["message"]["content"] or ""
        lat = time.time() - t0
    except Exception as e:
        print(f"{i+1:2}  {'ERR':<6} {stim[:45]}")
        print(f"   {e}")
        time.sleep(2)
        continue

    # Strip markdown fences
    clean = raw.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.+\}", clean, re.DOTALL)
    if not match:
        match = re.search(r"\{.+\}", raw, re.DOTALL)

    if match:
        try:
            d = json.loads(match.group())
            cat = d.get("category", "?")
            sal = d.get("salience", "?")
            act = d.get("action", "?")
            cat_ok = cat in VALID_CATS
            act_ok = act in VALID_ACTS
            sal_ok = isinstance(sal, (int, float)) and 0 <= sal <= 1
            status = "OK" if (cat_ok and act_ok and sal_ok) else "FAIL"
            if status == "OK":
                ok += 1
            print(f"{i+1:2}  {status:<6} {cat:<14} {expected:<14} {sal!s:>5} {act:<10} {lat:.1f}s")
        except Exception as e:
            print(f"{i+1:2}  {'PARSE':<6} {e}")
    else:
        print(f"{i+1:2}  {'NO_JSON':<6} {repr(raw[:80])}")

    time.sleep(1)  # Rate limit courtesy

print(f"\n{'='*70}")
print(f"Results: {ok}/{len(TEST_INPUTS)} OK = {ok/len(TEST_INPUTS):.0%} pass rate")
