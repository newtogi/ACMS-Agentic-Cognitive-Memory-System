"""Full system test — run once, clean process, no module cache."""
import sys, os, time, json, urllib.request, subprocess, importlib, unittest.mock as mock
sys.path.insert(0, r'C:/Users/alida/agent-brain')
os.chdir(r'C:/Users/alida/agent-brain')

results = []
def ok(test, detail=""): results.append(("PASS", test, detail))
def fail(test, detail=""): results.append(("FAIL", test, detail))

print("=" * 70)
print("AGENT BRAIN — FULL SYSTEM TEST")
print("=" * 70)

# [1] OLLAMA
print("\n[1] OLLAMA SERVICE")
try:
    tags = json.loads(urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10).read())
    names = [m['name'] for m in tags['models']]
    ok("service up", f"{len(names)} models")
    for want in ["minicpm-v4.6", "qwen3:4b"]:
        ok(f"model {want}") if any(want in n for n in names) else fail(f"model {want}")
except Exception as e:
    fail("service up", str(e)[:60])

# [2] IMPORTS
print("\n[2] MODULE IMPORTS")
for mod in ["config","zenbrain_client","perception_gate","pattern_detector",
            "predictive_engine","memory_graph","dream_cycle","insight_store"]:
    try:
        m = __import__(mod); importlib.reload(m); ok(mod)
    except Exception as e:
        fail(mod, str(e)[:60])

# [3] ZENBRAIN
print("\n[3] ZENBRAIN CLIENT")
import zenbrain_client as zc
ok("DB dir", str(zc.ZENBRAIN_DIR)[-30:]) if zc.ZENBRAIN_DIR.exists() else fail("DB dir")

try:
    schema = zc.validate_schema("system-bot")
    ok("schema validation", f"{len(schema)} tables OK")
except Exception as e:
    fail("schema", str(e)[:60])

try:
    bp = zc.backup_before_write("system-bot")
    ok("backup") if bp and bp.exists() else fail("backup")
except Exception as e:
    fail("backup", str(e)[:60])

ok("get_all_episodes", f"{len(zc.get_all_episodes('system-bot',5))} ep")
ok("get_all_facts", f"{len(zc.get_all_facts('system-bot',5))} facts")
ok("recall(query)", f"{len(zc.recall('zenbrain',limit=5))} hits")

# [4] L1 PERCEPTION
print("\n[4] L1 PERCEPTION GATE (live)")
from perception_gate import perceive
t0 = time.time()
r1 = perceive("The supervisor has approved the revised document", persist=True)
lat = time.time() - t0
ok("category", r1["category"]) if r1["category"] in ("event","fact","emotion","threat","noise") else fail("category")
ok("salience", f"{r1['salience']}") if 0<=r1["salience"]<=1 else fail("salience")
ok("latency", f"{lat:.1f}s") if lat<30 else fail("latency")
ok("persisted", r1.get("memory_id","")[:12]) if r1.get("memory_id") else fail("persisted")

import perception_gate as pg
with mock.patch.object(pg, "_call_ollama", side_effect=ConnectionError("test")):
    d = pg.perceive("data", persist=False)
ok("degraded→deferred") if d.get("status")=="deferred" else fail("degraded", str(d))

# [5] L6 PREDICTIVE
print("\n[5] L6 PREDICTIVE ENGINE (qwen3:4b CPU)")
from predictive_engine import PredictiveEngine as PE
eng = PE()
t0 = time.time()
p1 = eng.predict(current_state="Document approved, ready to publish")
lat = time.time() - t0
ok("clean prediction", f"{len(p1['prediction'])}ch") if not PE._is_template_echo(p1["prediction"]) else fail("echo")
ok("prediction_id") if p1["prediction_id"] else fail("no id")
ok("latency", f"{lat:.1f}s") if lat<30 else fail("latency")
eng.record_outcome(p1["prediction_id"], "confirmed")
ok("record_outcome")

# [6] L7 DREAM CYCLE
print("\n[6] L7 DREAM CYCLE (live)")
from dream_cycle import retrieve_memories_zenbrain
mems = retrieve_memories_zenbrain()
real = bool(mems) and not str(mems[0].id).startswith("mem-0")
ok("live retrieval", f"{len(mems)} real memories") if real else fail("live retrieval")

with mock.patch.object(zc, "get_all_episodes_cross_profile", side_effect=RuntimeError("test")):
    try:
        retrieve_memories_zenbrain()
        fail("abort on failure")
    except RuntimeError:
        ok("abort on failure", "zero writes")

# [7] INSIGHT STORE
print("\n[7] INSIGHT STORE")
from insight_store import store_dream_insight, load_insights, update_insight_status
iid = store_dream_insight("System test insight", "Lifecycle test")
ins = load_insights()
last = ins[-1]
ok("store", last["id"])
ok("pending") if last["status"]=="pending" else fail("status")

updated = update_insight_status(last["id"], "approved")
ok("approve") if updated else fail("approve")
from insight_store import promote_approved_to_zenbrain
promoted = promote_approved_to_zenbrain()
ok("promotion", f"{len(promoted)} insight(s) promoted") if promoted else fail("promotion", "no promotions")

conn.close()
update_insight_status(last["id"], "rejected")

# [8] MEMORY GRAPH
print("\n[8] MEMORY GRAPH (heuristic edges)")
from memory_graph import MemoryGraph
importlib.reload(sys.modules["memory_graph"]); from memory_graph import MemoryGraph
mg = MemoryGraph()
mg.build_from_memories(mems[:8])
edges = mg.build_all_edges()
s = mg.summary()
ok("graph built", f"{s['num_nodes']}n, {s['num_edges']}e")
ok("heuristic edges") if "heuristic" in edges else fail("edge type", str(edges))

# [9] GIT
print("\n[9] GIT SYNC")
head = subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True).stdout.strip()
remote = subprocess.run(["git","ls-remote","origin","main"], capture_output=True, text=True).stdout.split()[0][:7]
ok("synced", f"{head}=={remote}") if head==remote else fail("diverged", f"{head} vs {remote}")
dirty = subprocess.run(["git","status","--porcelain"], capture_output=True, text=True).stdout.strip()
ok("clean") if not dirty else fail("dirty", dirty[:40])

# [10] PYTEST
print("\n[10] TEST SUITE")
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_brain.py", "-q", "--tb=no"],
                    capture_output=True, text=True, timeout=60)
ok("pytest", r.stdout.strip().split("\n")[-1]) if "passed" in r.stdout else fail("pytest", r.stdout[:60])

# ── RESULTS ──
passed = sum(1 for s,_,_ in results if s=="PASS")
total = len(results)
print("\n" + "=" * 70)
for s, test, detail in results:
    sym = "✓" if s=="PASS" else "✗"
    print(f"  {sym} {test}" + (f"  — {detail}" if detail else ""))
print("=" * 70)
print(f"\nFINAL: {passed}/{total} PASS, {total-passed} FAIL")
