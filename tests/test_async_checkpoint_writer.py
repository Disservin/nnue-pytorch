import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import torch

import trainer.checkpoint_writer as cw
from trainer.callbacks import CheckpointManager
from trainer.checkpoint_writer import (
    AsyncCheckpointWriter,
    _atomic_save,
    cpu_snapshot,
)
from trainer.engine import SimpleTrainer

_ORIG_SAVE = torch.save
_ORIG_ATOMIC = cw._atomic_save


def _ckpt(tag, n=1000):
    return {
        "epoch": tag,
        "state_dict": {"w": torch.arange(n, dtype=torch.float32)},
        "blob": b"\x01" * 32,
        "nested": [{"x": torch.ones(4)}, (torch.zeros(2), 3)],
    }


class TestAsyncCheckpointWriter(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ckpt_writer_test_")
        self.path = os.path.join(self.dir, "last.ckpt")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_roundtrip(self):
        w = AsyncCheckpointWriter()
        ck = _ckpt(7)
        w.submit(ck, self.path)
        w.flush()
        loaded = torch.load(self.path, weights_only=False)
        self.assertEqual(loaded["epoch"], 7)
        self.assertTrue(torch.equal(loaded["state_dict"]["w"], ck["state_dict"]["w"]))
        self.assertEqual(loaded["blob"], ck["blob"])
        self.assertTrue(torch.equal(loaded["nested"][0]["x"], torch.ones(4)))
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_pool_reuse_and_isolation(self):
        """Repeated submits reuse the same buffers without cross-talk."""
        w = AsyncCheckpointWriter()
        w.submit(_ckpt(1), self.path)
        w.flush()
        loaded = torch.load(self.path, weights_only=False)
        ref = loaded["state_dict"]["w"].clone()
        # Mutate the source between epochs; the pooled snapshot must be
        # independent of it.
        src = _ckpt(2)
        src["state_dict"]["w"] = ref * 2
        w.submit(src, self.path)
        w.flush()
        loaded = torch.load(self.path, weights_only=False)
        self.assertTrue(torch.equal(loaded["state_dict"]["w"], ref * 2))
        self.assertEqual(len(w._pool), 1, "pool must not grow for same-shaped ckpts")
        self.assertEqual(w._free, {0})

    def test_coalescing_same_path(self):
        writes = []
        entered = threading.Event()
        release = threading.Event()

        def gated_atomic(ckpt, path):
            writes.append(os.path.basename(path))
            entered.set()
            release.wait(5)
            _ORIG_ATOMIC(ckpt, path)

        cw._atomic_save = gated_atomic
        try:
            w = AsyncCheckpointWriter()
            other = os.path.join(self.dir, "epoch=19.ckpt")
            w.submit(_ckpt(19), other)
            self.assertTrue(entered.wait(5))  # writer inside the slow write
            # First last.ckpt submit queues; the second must replace it
            # (same path) and reuse the freed buffer set.
            w.submit(_ckpt(1), self.path)
            w.submit(_ckpt(2), self.path)
            release.set()
            w.flush()
        finally:
            cw._atomic_save = _ORIG_ATOMIC
        self.assertEqual(
            torch.load(self.path, weights_only=False)["epoch"], 2,
            "the newest checkpoint must win",
        )
        last_writes = [p for p in writes if p.startswith("last")]
        self.assertLessEqual(len(last_writes), 2, f"too many last.ckpt writes: {writes}")
        self.assertEqual(
            torch.load(other, weights_only=False)["epoch"], 19,
            "distinct paths must not be dropped by coalescing",
        )
        self.assertLessEqual(len(w._pool), 2, "pool must stay bounded")

    def test_distinct_paths_not_dropped(self):
        w = AsyncCheckpointWriter()
        p2 = os.path.join(self.dir, "epoch=19-step=20480.ckpt")
        w.submit(_ckpt(19), p2)
        w.submit(_ckpt(20), self.path)
        w.flush()
        self.assertEqual(torch.load(p2, weights_only=False)["epoch"], 19)
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 20)

    def test_alias_roundtrip_and_history_isolation(self):
        """Periodic + last need one snapshot/write, and history stays immutable."""
        writer = AsyncCheckpointWriter()
        periodic = os.path.join(self.dir, "epoch=0.ckpt")
        with mock.patch.object(cw, "_atomic_save", wraps=_ORIG_ATOMIC) as save:
            writer.submit(_ckpt(1), periodic, aliases=(self.path,))
            writer.flush()
            self.assertEqual(save.call_count, 1)
        self.assertEqual(len(writer._pool), 1)
        self.assertTrue(os.path.samefile(periodic, self.path))
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 1)
        writer.submit(_ckpt(2), self.path)
        writer.flush()
        self.assertFalse(os.path.samefile(periodic, self.path))
        self.assertEqual(torch.load(periodic, weights_only=False)["epoch"], 1)
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 2)

    def test_manager_periodic_and_last_roundtrip(self):
        for asynchronous in (False, True):
            with self.subTest(asynchronous=asynchronous):
                directory = os.path.join(self.dir, str(asynchronous))
                model = torch.nn.Linear(2, 1)
                writer = AsyncCheckpointWriter() if asynchronous else None
                manager = CheckpointManager(
                    every_n_epochs=2, save_top_k=1, dirpath=directory,
                )
                trainer = SimpleTrainer(
                    model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                    schedulers=[], max_epochs=4, check_val_every_n_epoch=1,
                    gradient_clip_val=0, log_every_n_steps=1,
                    default_root_dir=directory, callbacks=[manager], logger=None,
                    device="cpu", rank=0, world_size=1, local_rank=0,
                    checkpoint_writer=writer,
                )
                with mock.patch.object(cw, "_atomic_save", wraps=_ORIG_ATOMIC) as save:
                    for epoch in range(4):
                        trainer.current_epoch = epoch
                        trainer.global_step = epoch + 1
                        with torch.no_grad():
                            model.weight.fill_(epoch)
                        manager.on_train_epoch_end(trainer)
                        trainer.flush_checkpoints()
                        last = os.path.join(directory, "last.ckpt")
                        checkpoint = torch.load(last, weights_only=False)
                        self.assertEqual(checkpoint["epoch"], epoch)
                        self.assertEqual(checkpoint["global_step"], epoch + 1)
                        self.assertTrue(torch.all(checkpoint["state_dict"]["weight"] == epoch))
                        if (epoch + 1) % 2 == 0:
                            periodic = os.path.join(directory, f"epoch={epoch}-step={epoch + 1}.ckpt")
                            self.assertTrue(os.path.samefile(last, periodic))
                    manager.on_train_end(trainer)
                    self.assertEqual(save.call_count, 4)
                self.assertFalse(os.path.exists(os.path.join(directory, "epoch=1-step=2.ckpt")))
                self.assertTrue(os.path.exists(os.path.join(directory, "epoch=3-step=4.ckpt")))

    def test_alias_copy_fallback_with_stale_link(self):
        periodic = os.path.join(self.dir, "epoch=0.ckpt")
        _atomic_save(_ckpt(0), periodic)
        os.link(periodic, self.path + ".tmp")
        new_periodic = os.path.join(self.dir, "epoch=1.ckpt")
        with mock.patch.object(cw.os, "link", side_effect=OSError("not supported")):
            cw._save_with_aliases(_ckpt(1), new_periodic, (self.path,))
        self.assertEqual(torch.load(periodic, weights_only=False)["epoch"], 0)
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 1)
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_alias_atomic_on_failure(self):
        _atomic_save(_ckpt(0), self.path)
        periodic = os.path.join(self.dir, "epoch=1.ckpt")
        original_replace = cw.os.replace

        def fail_alias(src, dst):
            if dst == self.path:
                raise OSError("alias publication failed")
            return original_replace(src, dst)

        with (
            mock.patch.object(cw.os, "replace", side_effect=fail_alias),
            self.assertRaisesRegex(OSError, "alias publication failed"),
        ):
            cw._save_with_aliases(_ckpt(1), periodic, (self.path,))
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 0)
        self.assertEqual(torch.load(periodic, weights_only=False)["epoch"], 1)
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_alias_survives_pruned_periodic(self):
        periodic = os.path.join(self.dir, "epoch=0.ckpt")
        original_replace = cw.os.replace

        def prune_periodic(src, dst):
            original_replace(src, dst)
            if dst == periodic:
                os.remove(periodic)

        with mock.patch.object(cw.os, "replace", side_effect=prune_periodic):
            cw._save_with_aliases(_ckpt(1), periodic, (self.path,))
        self.assertFalse(os.path.exists(periodic))
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 1)
        self.assertFalse(os.path.exists(periodic + ".tmp"))

    def test_sync_fallback_waits_for_older_alias(self):
        """An in-flight periodic alias cannot overwrite a newer sync last save."""
        writer = AsyncCheckpointWriter()
        entered = threading.Event()
        release = threading.Event()
        waiting = threading.Event()
        original_wait = writer._cv.wait_for

        def gated_save(ckpt, path):
            if ckpt["epoch"] == 0:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test writer gate timed out")
            _ORIG_ATOMIC(ckpt, path)

        def wait_for(*args, **kwargs):
            waiting.set()
            return original_wait(*args, **kwargs)

        errors = []
        def save_latest():
            try:
                writer.submit(_ckpt(2), self.path)
            except Exception as error:  # noqa: BLE001 -- report thread errors to the test
                errors.append(error)

        with mock.patch.object(cw, "_atomic_save", side_effect=gated_save):
            periodic = os.path.join(self.dir, "epoch=0.ckpt")
            writer.submit(_ckpt(0), periodic, aliases=(self.path,))
            self.assertTrue(entered.wait(5))
            writer.submit(_ckpt(1), os.path.join(self.dir, "epoch=1.ckpt"))
            with mock.patch.object(writer._cv, "wait_for", side_effect=wait_for):
                thread = threading.Thread(target=save_latest, daemon=True)
                thread.start()
                try:
                    self.assertTrue(waiting.wait(5))
                finally:
                    release.set()
                    thread.join(5)
                self.assertFalse(thread.is_alive())
                writer.flush()
        self.assertFalse(errors)
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 2)
        self.assertEqual(torch.load(periodic, weights_only=False)["epoch"], 0)

    def test_atomic_on_failure(self):
        _atomic_save(_ckpt(1), self.path)
        with open(self.path, "rb") as f:
            good = f.read()

        def boom(obj, path):
            with open(path, "wb") as fh:
                fh.write(b"partial")
            raise RuntimeError("disk full")

        cw.torch.save = boom
        try:
            w = AsyncCheckpointWriter()
            w.submit(_ckpt(2), self.path)
            with self.assertRaises(RuntimeError):
                w.flush()
        finally:
            cw.torch.save = _ORIG_SAVE
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), good, "old file must survive")
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_fallback_to_sync_after_failure(self):
        def boom(obj, path):
            raise RuntimeError("fail")

        cw._atomic_save = boom
        try:
            w = AsyncCheckpointWriter()
            w.submit(_ckpt(1), self.path)
            time.sleep(0.2)
            cw._atomic_save = _ORIG_ATOMIC
            w.submit(_ckpt(2), self.path)  # must write synchronously
        finally:
            cw._atomic_save = _ORIG_ATOMIC
        self.assertEqual(torch.load(self.path, weights_only=False)["epoch"], 2)
        with self.assertRaises(RuntimeError):
            w.flush()  # recorded error still surfaces

    def test_cpu_snapshot_isolation(self):
        if torch.cuda.is_available():
            t = torch.ones(64, device="cuda")
            snap = cpu_snapshot({"t": t})
            t.add_(1.0)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(snap["t"], torch.ones(64)))
            self.assertEqual(snap["t"].device.type, "cpu")
        else:
            t = torch.ones(64)
            snap = cpu_snapshot({"t": t})
            t.add_(1.0)
            self.assertTrue(torch.equal(snap["t"], torch.ones(64)))


if __name__ == "__main__":
    unittest.main()
