"""`evo asset` CLI round-trip against a temp workspace (#55)."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from unittest import mock

from evo.assets import assets_path, load_registry
from evo.cli import (
    cmd_asset_get,
    cmd_asset_list,
    cmd_asset_put,
    cmd_asset_rm,
    cmd_asset_use,
)
from evo.core import init_workspace


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@evo"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=root, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=root, check=True)
    (root / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)


def _put_args(path, name, kind, exp=None, tag=None, copy=False, backend=None):
    return argparse.Namespace(path=str(path), name=name, kind=kind, exp=exp,
                              tag=tag or [], copy=copy, backend=backend)


class _FakeRemoteBackend:
    """In-memory stand-in for S3/HF: upload stores bytes by uri, download
    writes them into dest_dir under the uri's basename."""
    _store: dict = {}

    def upload(self, local, uri):
        type(self)._store[uri] = Path(local).read_bytes()

    def download(self, uri, dest_dir):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        dest = Path(dest_dir) / uri.rstrip("/").split("/")[-1]
        dest.write_bytes(type(self)._store[uri])
        return dest

    def exists(self, uri):
        return uri in type(self)._store


class _FakeNestedBackend(_FakeRemoteBackend):
    """Like huggingface_hub(local_dir=...): keeps the repo subpath under dest_dir
    (the flat fake above hides that), and counts downloads."""
    downloads = 0

    def download(self, uri, dest_dir):
        type(self).downloads += 1
        rel = uri.split("://", 1)[1].split("/", 2)[2]  # drop owner/name
        dest = Path(dest_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(type(self)._store[uri])
        return dest


class TestAssetCli(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        _init_git_repo(self.root)
        init_workspace(self.root, target="t.py", benchmark="python bench.py",
                       metric="max", gate=None)
        # A file to register as an asset.
        self.asset_file = self.root / "adapter.bin"
        self.asset_file.write_text("weights")
        self._old_cwd = Path.cwd()
        os.chdir(self.root)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def _capture(self, fn, args) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(args)
        return buf.getvalue().strip()

    def test_put_registers_asset(self):
        cmd_asset_put(_put_args(self.asset_file, "adapter", "checkpoint",
                                exp="exp_0001", tag=["epoch=2"]))
        reg = load_registry(self.root)
        entry = reg["assets"]["adapter"]
        self.assertEqual(entry["kind"], "checkpoint")
        self.assertEqual(entry["produced_by"], "exp_0001")
        self.assertEqual(entry["tags"], {"epoch": "2"})
        self.assertEqual(Path(entry["path"]).read_text(), "weights")

    def test_put_missing_path_errors(self):
        with self.assertRaises(RuntimeError):
            cmd_asset_put(_put_args(self.root / "nope.bin", "x", "model"))

    def test_put_duplicate_name_errors(self):
        cmd_asset_put(_put_args(self.asset_file, "adapter", "checkpoint"))
        with self.assertRaises(RuntimeError):
            cmd_asset_put(_put_args(self.asset_file, "adapter", "model"))

    def test_get_prints_path(self):
        cmd_asset_put(_put_args(self.asset_file, "adapter", "checkpoint"))
        out = self._capture(cmd_asset_get, argparse.Namespace(name="adapter"))
        self.assertEqual(out, str(self.asset_file))

    def test_get_unknown_raises(self):
        with self.assertRaises(RuntimeError):
            cmd_asset_get(argparse.Namespace(name="ghost"))

    def test_put_copy_materializes_under_assets_dir(self):
        cmd_asset_put(_put_args(self.asset_file, "adapter", "checkpoint", copy=True))
        entry = load_registry(self.root)["assets"]["adapter"]
        self.assertTrue(entry["copied"])
        self.assertIn("assets", Path(entry["path"]).parts)
        self.assertEqual(Path(entry["path"]).read_text(), "weights")

    def test_list_filters_by_tag(self):
        cmd_asset_put(_put_args(self.asset_file, "a", "dataset", tag=["held-out=true"]))
        cmd_asset_put(_put_args(self.asset_file, "b", "dataset"))
        out = self._capture(cmd_asset_list, argparse.Namespace(
            kind=None, tag=["held-out=true"], produced_by=None, consumed_by=None, json=True))
        names = [e["name"] for e in json.loads(out)]
        self.assertEqual(names, ["a"])

    def test_use_records_consumption(self):
        cmd_asset_put(_put_args(self.asset_file, "adapter", "checkpoint"))
        cmd_asset_use(argparse.Namespace(name="adapter", exp="exp_0002"))
        self.assertEqual(
            load_registry(self.root)["assets"]["adapter"]["consumed_by"], ["exp_0002"])

    def test_rm_refuses_when_consumed(self):
        cmd_asset_put(_put_args(self.asset_file, "adapter", "checkpoint"))
        cmd_asset_use(argparse.Namespace(name="adapter", exp="exp_0002"))
        with self.assertRaises(RuntimeError):
            cmd_asset_rm(argparse.Namespace(name="adapter", force=False))
        cmd_asset_rm(argparse.Namespace(name="adapter", force=True))
        self.assertNotIn("adapter", load_registry(self.root)["assets"])

    def test_put_normalizes_whitespace_name(self):
        # A padded handle must register under the trimmed name and be reachable
        # both by the trimmed name and by the padded string the user typed.
        cmd_asset_put(_put_args(self.asset_file, "  spaced  ", "checkpoint"))
        entry = load_registry(self.root)["assets"]["spaced"]
        self.assertEqual(entry["name"], "spaced")
        self.assertEqual(
            self._capture(cmd_asset_get, argparse.Namespace(name="spaced")),
            str(self.asset_file))
        self.assertEqual(
            self._capture(cmd_asset_get, argparse.Namespace(name="  spaced  ")),
            str(self.asset_file))

    def test_put_whitespace_name_duplicate_detected(self):
        cmd_asset_put(_put_args(self.asset_file, "spaced", "checkpoint"))
        with self.assertRaises(RuntimeError):
            cmd_asset_put(_put_args(self.asset_file, "  spaced  ", "model"))

    def test_put_blank_name_rejected(self):
        with self.assertRaises(RuntimeError):
            cmd_asset_put(_put_args(self.asset_file, "   ", "model"))

    def test_registry_file_location(self):
        cmd_asset_put(_put_args(self.asset_file, "adapter", "checkpoint"))
        self.assertTrue(assets_path(self.root).exists())

    # --- storage backends (#55 follow-up) -------------------------------

    def test_put_backend_uploads_and_records_uri(self):
        _FakeRemoteBackend._store = {}
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            cmd_asset_put(_put_args(self.asset_file, "remote-adapter", "checkpoint",
                                    backend="s3://bucket/models/remote-adapter.bin"))
        entry = load_registry(self.root)["assets"]["remote-adapter"]
        self.assertEqual(entry["backend"], "s3")
        self.assertEqual(entry["uri"], "s3://bucket/models/remote-adapter.bin")
        self.assertIn("s3://bucket/models/remote-adapter.bin", _FakeRemoteBackend._store)

    def test_get_remote_downloads_to_cache(self):
        _FakeRemoteBackend._store = {}
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            cmd_asset_put(_put_args(self.asset_file, "remote-adapter", "checkpoint",
                                    backend="s3://bucket/models/remote-adapter.bin"))
            out = self._capture(cmd_asset_get, argparse.Namespace(name="remote-adapter"))
        # get returns a LOCAL cache path whose contents match the uploaded file.
        self.assertTrue(Path(out).exists())
        self.assertEqual(Path(out).read_text(), "weights")
        self.assertIn("_cache", Path(out).parts)

    def test_local_put_records_local_backend(self):
        cmd_asset_put(_put_args(self.asset_file, "adapter", "checkpoint"))
        entry = load_registry(self.root)["assets"]["adapter"]
        self.assertEqual(entry.get("backend", "local"), "local")

    def test_backend_plain_path_get_returns_local_path(self):
        # A plain-path --backend records backend="local" with path=None; get must
        # still resolve to a real local file, not print "None".
        store = self.root / "store" / "adapter.bin"
        cmd_asset_put(_put_args(self.asset_file, "adapter", "checkpoint",
                                backend=str(store)))
        out = self._capture(cmd_asset_get, argparse.Namespace(name="adapter"))
        self.assertNotEqual(out, "None")
        self.assertEqual(Path(out).read_text(), "weights")

    def test_list_and_use_show_uri_for_remote_asset(self):
        _FakeRemoteBackend._store = {}
        uri = "s3://bucket/models/remote-adapter.bin"
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            cmd_asset_put(_put_args(self.asset_file, "remote-adapter", "checkpoint",
                                    backend=uri))
        listed = self._capture(cmd_asset_list, argparse.Namespace(
            kind=None, tag=[], produced_by=None, consumed_by=None, json=False))
        self.assertIn(uri, listed)
        self.assertNotIn("None", listed)
        used = self._capture(cmd_asset_use, argparse.Namespace(
            name="remote-adapter", exp="exp_0002"))
        self.assertIn(uri, used)
        self.assertNotIn("None", used)

    def test_get_nested_remote_path_reuses_cache(self):
        # Backends that keep the remote subpath (HF) must still hit the cache on
        # the second get instead of re-downloading every time.
        _FakeNestedBackend._store = {}
        _FakeNestedBackend.downloads = 0
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeNestedBackend()):
            cmd_asset_put(_put_args(self.asset_file, "nested", "checkpoint",
                                    backend="hf://org/model/ckpt/epoch2/a.bin"))
            first = self._capture(cmd_asset_get, argparse.Namespace(name="nested"))
            second = self._capture(cmd_asset_get, argparse.Namespace(name="nested"))
        self.assertEqual(first, second)
        self.assertEqual(Path(second).read_text(), "weights")
        self.assertEqual(_FakeNestedBackend.downloads, 1)

    def test_put_backend_validates_before_uploading(self):
        _FakeRemoteBackend._store = {}
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            with self.assertRaises((ValueError, RuntimeError)):
                cmd_asset_put(_put_args(self.asset_file, "x", "  ",
                                        backend="s3://bucket/x.bin"))
        self.assertEqual(_FakeRemoteBackend._store, {})

    def test_put_backend_rejects_directory(self):
        _FakeRemoteBackend._store = {}
        d = self.root / "somedir"
        d.mkdir()
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            with self.assertRaises(RuntimeError):
                cmd_asset_put(_put_args(d, "dirasset", "dataset",
                                        backend="s3://bucket/d"))
        self.assertEqual(_FakeRemoteBackend._store, {})

    def test_get_does_not_serve_stale_cache_after_repoint(self):
        # rm + put the same handle at a different uri with the same basename:
        # get must fetch the new bytes, not the previous uri's cached copy.
        _FakeRemoteBackend._store = {}
        new_file = self.root / "new.bin"
        new_file.write_text("new-weights")
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            cmd_asset_put(_put_args(self.asset_file, "ckpt", "checkpoint",
                                    backend="s3://run-a/model.bin"))
            first = self._capture(cmd_asset_get, argparse.Namespace(name="ckpt"))
            first_bytes = Path(first).read_text()  # rm below drops this cached copy
            cmd_asset_rm(argparse.Namespace(name="ckpt", force=False))
            cmd_asset_put(_put_args(new_file, "ckpt", "checkpoint",
                                    backend="s3://run-b/model.bin"))
            second = self._capture(cmd_asset_get, argparse.Namespace(name="ckpt"))
        self.assertEqual(first_bytes, "weights")
        self.assertEqual(Path(second).read_text(), "new-weights")

    def test_rm_then_reput_same_uri_serves_fresh_bytes(self):
        # Iterating on a checkpoint at a fixed key: rm + put uploads new bytes to
        # the SAME uri, so a cache keyed only by uri would still serve the old file.
        _FakeRemoteBackend._store = {}
        uri = "s3://bucket/best.bin"
        new_file = self.root / "newer.bin"
        new_file.write_text("newer-weights")
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            cmd_asset_put(_put_args(self.asset_file, "best", "checkpoint", backend=uri))
            first = self._capture(cmd_asset_get, argparse.Namespace(name="best"))
            cmd_asset_rm(argparse.Namespace(name="best", force=False))
            self.assertFalse(Path(first).exists())  # rm drops the downloaded copy
            cmd_asset_put(_put_args(new_file, "best", "checkpoint", backend=uri))
            second = self._capture(cmd_asset_get, argparse.Namespace(name="best"))
        self.assertEqual(Path(second).read_text(), "newer-weights")

    def test_rm_keeps_entry_when_cache_cannot_be_cleared(self):
        # If the cached copy can't be deleted (e.g. Windows file lock), rm must
        # fail with the entry still registered; dropping it would leave stale
        # bytes that a later put at the same uri would serve.
        _FakeRemoteBackend._store = {}
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            cmd_asset_put(_put_args(self.asset_file, "held", "checkpoint",
                                    backend="s3://bucket/held.bin"))
            self._capture(cmd_asset_get, argparse.Namespace(name="held"))
        with mock.patch("evo.assets.shutil.rmtree", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                cmd_asset_rm(argparse.Namespace(name="held", force=False))
        self.assertIn("held", load_registry(self.root)["assets"])

    def test_rm_local_asset_without_cache_succeeds(self):
        # Local assets never populate a cache; clearing a missing dir is a no-op.
        cmd_asset_put(_put_args(self.asset_file, "plainlocal", "checkpoint"))
        cmd_asset_rm(argparse.Namespace(name="plainlocal", force=False))
        self.assertNotIn("plainlocal", load_registry(self.root)["assets"])

    def test_put_backend_uploads_outside_registry_lock(self):
        # advisory_lock gives up after 10s; a big upload under it would fail every
        # concurrent `evo asset` call. Upload must not hold the registry lock.
        from evo.core import lock_file_for
        from evo.locking import advisory_lock

        root = self.root
        held = []

        class _ProbingBackend(_FakeRemoteBackend):
            def upload(self, local, uri):
                try:
                    with advisory_lock(lock_file_for(assets_path(root)),
                                       timeout_seconds=0.3):
                        held.append(False)
                except Exception:
                    held.append(True)
                super().upload(local, uri)

        _FakeRemoteBackend._store = {}
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_ProbingBackend()):
            cmd_asset_put(_put_args(self.asset_file, "big", "checkpoint",
                                    backend="s3://bucket/big.bin"))
        self.assertEqual(held, [False])
        self.assertIn("big", load_registry(self.root)["assets"])

    def test_put_backend_taken_name_does_not_upload(self):
        # A taken name must be rejected before the upload can overwrite a remote
        # object.
        _FakeRemoteBackend._store = {}
        cmd_asset_put(_put_args(self.asset_file, "taken", "model"))
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            with self.assertRaises(RuntimeError):
                cmd_asset_put(_put_args(self.asset_file, "taken", "model",
                                        backend="s3://bucket/taken.bin"))
        self.assertEqual(_FakeRemoteBackend._store, {})

    def test_put_backend_env_var_collision_does_not_upload(self):
        # 'a_b' would shadow 'a-b' (both EVO_ASSET_A_B); that must be refused
        # before the upload, not after it leaves an orphaned remote object.
        _FakeRemoteBackend._store = {}
        cmd_asset_put(_put_args(self.asset_file, "a-b", "model"))
        with mock.patch("evo.asset_backends.backend_for_uri",
                        return_value=_FakeRemoteBackend()):
            with self.assertRaises((ValueError, RuntimeError)):
                cmd_asset_put(_put_args(self.asset_file, "a_b", "model",
                                        backend="s3://bucket/a_b.bin"))
        self.assertEqual(_FakeRemoteBackend._store, {})
        self.assertNotIn("a_b", load_registry(self.root)["assets"])

    def test_put_rejects_path_separators_in_name(self):
        # The name becomes a directory under the workspace (--copy / remote
        # cache), so it must not be able to escape it.
        for bad in ("../evil", "a/b", "a\\b", "..", "."):
            with self.assertRaises(RuntimeError, msg=bad):
                cmd_asset_put(_put_args(self.asset_file, bad, "model", copy=True))


if __name__ == "__main__":
    unittest.main()
