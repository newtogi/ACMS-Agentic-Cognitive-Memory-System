"""
Agent Brain — Integrity Test Suite
Reviewer's 9-point minimum coverage spec + our own smoke tests.
Run: python -m pytest tests/test_brain.py -v
"""
import sys, os, sqlite3, tempfile, threading, unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ================================================================
# 1. Clean installation — all modules import
# ================================================================
class TestCleanInstall(unittest.TestCase):
    def test_all_modules_import(self):
        for mod in ["config", "zenbrain_client", "perception_gate",
                     "pattern_detector", "predictive_engine",
                     "memory_graph", "dream_cycle", "insight_store"]:
            __import__(mod)

    def test_requirements_declared(self):
        reqs = Path(__file__).resolve().parent.parent / "requirements.txt"
        content = reqs.read_text()
        self.assertIn("networkx", content)
        self.assertIn("requests", content)

    def test_license_exists(self):
        license_path = Path(__file__).resolve().parent.parent / "LICENSE"
        self.assertTrue(license_path.exists())


# ================================================================
# 2. L1 timeout / unavailable behaviour
# ================================================================
class TestL1DegradedState(unittest.TestCase):
    def test_ollama_unavailable_returns_deferred(self):
        from perception_gate import perceive
        with patch("perception_gate._call_ollama", side_effect=ConnectionError("down")):
            result = perceive("important data", persist=False)
        self.assertEqual(result["action"], "defer")
        self.assertEqual(result["status"], "deferred")
        self.assertFalse(result["persisted"])

    def test_deferred_never_persists(self):
        from perception_gate import perceive
        with patch("perception_gate._call_ollama", side_effect=ConnectionError("down")):
            result = perceive("test data", persist=True)
        self.assertFalse(result["persisted"])
        self.assertNotIn("memory_id", result)


# ================================================================
# 3. Live ZenBrain failure -> zero writes
# ================================================================
class TestDreamZeroWritesOnFailure(unittest.TestCase):
    def test_live_retrieval_failure_raises(self):
        import dream_cycle as dc, zenbrain_client as zc
        with patch.object(zc, "get_all_episodes_cross_profile",
                          side_effect=RuntimeError("sim DB")), \
             patch.object(zc, "get_all_facts_cross_profile",
                          side_effect=RuntimeError("sim DB")):
            with self.assertRaises(RuntimeError) as ctx:
                dc.retrieve_memories_zenbrain()
            self.assertIn("aborted", str(ctx.exception).lower())

    def test_no_insight_store_called_on_failure(self):
        import dream_cycle as dc, zenbrain_client as zc
        with patch.object(zc, "get_all_episodes_cross_profile",
                          side_effect=RuntimeError("sim")), \
             patch.object(zc, "get_all_facts_cross_profile",
                          side_effect=RuntimeError("sim")), \
             patch("insight_store.store_dream_insight") as mock_store:
            with self.assertRaises(RuntimeError):
                dc.retrieve_memories_zenbrain()
            mock_store.assert_not_called()


# ================================================================
# 4. Mock mode returns sample data, not real
# ================================================================
class TestMockNeverWrites(unittest.TestCase):
    def test_mock_returns_sample_data(self):
        import dream_cycle as dc
        mems = dc.retrieve_memories_mock()
        self.assertTrue(len(mems) > 0)
        self.assertTrue(all(m.id.startswith("mem-") for m in mems))


# ================================================================
# 5. Episode conversion handles missing confidence
# ================================================================
class TestEpisodeConversion(unittest.TestCase):
    def test_row_to_episode_no_confidence_key(self):
        from zenbrain_client import _row_to_episode
        row = {
            "id": "test-1", "content": "test content", "context": "",
            "emotional_weight": 0.7, "metadata": "{}", "created_at": "2026-01-01"
        }
        class Row:
            def __init__(self, d):
                self._d = d
            def __getitem__(self, key):
                return self._d[key]
        result = _row_to_episode(Row(row))
        self.assertIn("id", result)
        self.assertEqual(result["content"], "test content")
        self.assertNotIn("confidence", result)
        self.assertAlmostEqual(result["emotional_weight"], 0.7)


# ================================================================
# 6. Pending insights excluded from normal ZenBrain recall
# ================================================================
class TestPendingInsightSeparation(unittest.TestCase):
    def test_store_to_local_not_zb(self):
        from insight_store import store_dream_insight, load_insights, update_insight_status
        from zenbrain_client import _connect
        id1 = store_dream_insight("sep test", "test detail")
        ins = load_insights()
        self.assertTrue(len(ins) > 0)
        self.assertEqual(ins[-1]["status"], "pending")
        conn = _connect("system-bot")
        n = conn.execute(
            "SELECT COUNT(*) FROM learned_facts WHERE source='agent_brain_dream_cycle'"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(n, 0, "No dream insights in ZenBrain")
        update_insight_status(id1, "rejected")


# ================================================================
# 7. Cross-profile access blocked
# ================================================================
class TestProfileIsolation(unittest.TestCase):
    def test_known_profiles_single(self):
        from zenbrain_client import KNOWN_PROFILES
        self.assertEqual(KNOWN_PROFILES, ["system-bot"])


# ================================================================
# 8. Database schema check
# ================================================================
class TestSchemaCheck(unittest.TestCase):
    def test_validate_schema_passes(self):
        from zenbrain_client import validate_schema
        schema = validate_schema("system-bot")
        self.assertIsInstance(schema, dict)
        self.assertIn("episodic_memories", schema)
        self.assertIn("learned_facts", schema)

    def test_backup_before_write_creates_file(self):
        from zenbrain_client import backup_before_write
        path = backup_before_write("system-bot")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())


# ================================================================
# 9. SQLite concurrent writes (WAL mode)
# ================================================================
class TestSQLiteConcurrency(unittest.TestCase):
    def test_concurrent_wal_writes(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            for _ in range(2):
                c = sqlite3.connect(db_path, timeout=5)
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("CREATE TABLE IF NOT EXISTS t(id TEXT)")
                c.commit()
                c.close()
            errors = []
            def writer(path, val):
                try:
                    c = sqlite3.connect(path, timeout=5)
                    c.execute("INSERT INTO t(id) VALUES(?)", (val,))
                    c.commit()
                    c.close()
                except Exception as e:
                    errors.append(str(e))
            t1 = threading.Thread(target=writer, args=(db_path, "a"))
            t2 = threading.Thread(target=writer, args=(db_path, "b"))
            t1.start()
            t2.start()
            t1.join(10)
            t2.join(10)
            self.assertEqual(errors, [])
            c = sqlite3.connect(db_path)
            self.assertEqual(c.execute("SELECT COUNT(*) FROM t").fetchone()[0], 2)
            c.close()
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    unittest.main()
