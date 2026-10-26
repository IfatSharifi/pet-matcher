"""Regression: the matches page loads several photos in parallel. A single shared SQLite
connection raised 'sqlite3.InterfaceError: bad parameter or other API misuse' (seen as a
500 on /photos/<id>/0 while recording the demo video)."""
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from petmatch.store import Report, Store


class TestStoreThreads(unittest.TestCase):
    def test_parallel_reads_and_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "t.db")
            try:
                ids = [store.add_report(Report(side="lost", species="dog", colors=["black"], event_date=date.today(),
                                               contact_phone="050"),
                                        [(f"p{i}_{k}.jpg", np.ones(8, np.float32)) for k in range(3)])
                       for i in range(20)]
                errors, wrong = [], []

                def reader(n):
                    try:
                        for _ in range(60):
                            rid = ids[(n * 7) % len(ids)]
                            paths = store.photo_paths(rid)
                            if len(paths) != 3 or not all(p.startswith(f"p{ids.index(rid)}_") for p in paths):
                                wrong.append((rid, paths))
                            store.get(rid)
                            store.open_reports("lost")
                    except Exception as e:  # noqa: BLE001
                        errors.append(repr(e))

                def writer():
                    try:
                        for _ in range(20):
                            store.add_report(Report(side="found", species="dog", colors=["tan"],
                                                    event_date=date.today(), contact_phone="050"))
                    except Exception as e:  # noqa: BLE001
                        errors.append(repr(e))

                threads = [threading.Thread(target=reader, args=(i,)) for i in range(12)]
                threads.append(threading.Thread(target=writer))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                self.assertEqual(errors, [])
                self.assertEqual(wrong, [])
                self.assertEqual(len(store.open_reports("found")), 20)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
