"""Full test of hybrid classifier against all 24 inputs."""
import sys, time
sys.path.insert(0, r"C:/Users/alida/agent-brain")
from hybrid_classifier import classify, get_stats, EXAMPLES

TEST_INPUTS = [
    # Easy cases
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
    # Edge cases
    ("The user is crying and said their dog died yesterday.", "emotional"),
    ("If the build fails, run the rollback script and notify the team.", "procedural"),
    ("The word 'bank' can mean a financial institution or a river edge.", "semantic"),
    ("User just said 'thanks for nothing' and left the chat.", "emotional"),
    ("Schedule a meeting about the meeting scheduling process.", "social"),
    ("The model's accuracy dropped from 94% to 67% after the update.", "procedural"),
    ("I don't know what to do next.", "episodic"),
    ("Server load is at 98% and climbing — need immediate action.", "perceptual"),
    ("The user asked me to forget everything they just said.", "episodic"),
    ("The user's name is pronounced 'ee-lah' not 'ah-li-da'.", "semantic"),
    ("This sentence is a test of the emergency broadcast system.", "noise"),
    ("The user wants me to never use the word 'moist' again.", "episodic"),
]

print("Hybrid Perception Gate — Full Test (24 inputs)\n")
print(f"{'#':>2}  {'Status':<6} {'Layer':<10} {'Cat':<14} {'Exp':<14} {'Sal':>5} {'Lat':>7}  Input")
print("-" * 95)

ok = 0
total_time = 0
layer_counts = {"embedding": 0, "llm": 0, "cache": 0}

for i, (stim, expected) in enumerate(TEST_INPUTS):
    result = classify(stim)
    cat = result.get("category", "?")
    sal = result.get("salience", 0)
    layer = result.get("layer", "?")
    lat = result.get("latency_ms", 0)

    status = "OK" if cat == expected else "DIFF"
    if status == "OK":
        ok += 1

    total_time += lat
    layer_counts[layer] = layer_counts.get(layer, 0) + 1

    print(f"{i+1:2}  {status:<6} {layer:<10} {cat:<14} {expected:<14} {sal:>5.2f} {lat:>5}ms  {stim[:45]}")

print(f"\n{'='*95}")
print(f"Results: {ok}/{len(TEST_INPUTS)} = {ok/len(TEST_INPUTS):.0%} accuracy")
print(f"Total time: {total_time}ms ({total_time/1000:.1f}s)")
print(f"Avg latency: {total_time/len(TEST_INPUTS):.0f}ms per input")
print(f"Layer usage: {layer_counts}")
print(f"Speedup vs LLM-only: {len(TEST_INPUTS)*5000/total_time:.1f}x")
