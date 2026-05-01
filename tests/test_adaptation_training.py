from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path
from unittest import mock

from PIL import Image

from agentos_orchestrator.cognition.adaptation_training import (
    AdaptationLongRunConfig,
    AdaptationTrainingConfig,
    UnknownAppAdaptationTrainer,
    _distributed_targets,
    _osworld_shard_batches,
)
from agentos_orchestrator.os_control.base import UiAction
from agentos_orchestrator.cognition.trajectory_recorder import TrajectoryRecorder
from agentos_orchestrator.cognition.runtime_state import OutcomeEvaluation
from agentos_orchestrator.cognition.abstract_world_model import AbstractUIState


class StubAdaptationTrainer(UnknownAppAdaptationTrainer):
    _image_bytes: bytes = b""

    def __init__(self, workspace_root: str | Path, image_bytes: bytes) -> None:
        super().__init__(workspace_root)
        type(self)._image_bytes = image_bytes

    @staticmethod
    def _fetch_json(url: str):
        if "datasets-server.huggingface.co/rows" in url:
            if "mlfoundations/Click-100k" in url:
                return {
                    "rows": [
                        {
                            "row": {
                                "image_path": "desktop/sample.png",
                                "images": [
                                    {
                                        "src": "stub://image",
                                        "width": 400,
                                        "height": 200,
                                    }
                                ],
                                "easyr1_prompt": "Context <image> Type hello in the search field.",
                                "bbox": [40, 20, 140, 60],
                                "image_width": 400,
                                "image_height": 200,
                            }
                        }
                    ]
                }
            if "cckevinn/GUI-Actor-Data/tree/main" in url:
                return []
            return {
                "rows": [
                    {
                        "row": {
                            "file_name": "sample.png",
                            "bbox": [0.1, 0.1, 0.4, 0.3],
                            "instruction": "search for recent files",
                            "data_type": "text",
                            "data_source": "windows",
                            "image": {
                                "src": "stub://image",
                                "width": 400,
                                "height": 200,
                            },
                        }
                    }
                ]
            }
        return {
            "libreoffice_writer": {"task-1": 1.0},
            "gimp": {"task-2": 0.0},
            "multi_apps": {"task-3": 1.0},
        }

    @staticmethod
    def _fetch_bytes(url: str) -> bytes:
        if url != "stub://image":
            raise AssertionError(f"Unexpected image URL: {url}")
        return StubAdaptationTrainer._image_bytes

    @staticmethod
    def fetch_json(url: str):
        return UnknownAppAdaptationTrainer._fetch_json(url)

    def gui_actor_training_sample(
        self,
        row: object,
        archive: zipfile.ZipFile,
        source_name: str,
    ):
        if not isinstance(row, dict):
            raise AssertionError("row must be a dict")
        return self._gui_actor_training_sample(
            row,
            archive,
            source_name,
            self._archive_image_prefixes(archive),
        )

    def iter_transition_candidates(self, payload: object):
        return self._iter_transition_candidates(payload)

    def _list_osworld_archives(self, *, limit: int) -> list[str]:
        archives = [
            "archive_00.zip",
            "archive_01.zip",
            "archive_02.zip",
            "archive_03.zip",
        ]
        return archives[:limit] if limit > 0 else archives

    def _train_world_model_from_osworld_archives(
        self,
        *,
        archive_limit: int,
        transition_limit: int,
        archive_paths: list[str] | None,
        cache_root: Path | None,
        cache_budget_bytes: int,
        stage_remote_archives: bool,
        output_path: Path,
    ) -> dict[str, int]:
        paths = archive_paths or self._list_osworld_archives(limit=archive_limit)
        transitions = transition_limit if transition_limit > 0 else len(paths)
        output_path.write_text(
            json.dumps(
                {
                    "archive": paths[0] if paths else "stub-osworld.zip",
                    "before": {"app_context": "browser"},
                    "action": {"action_type": "click", "selector": "name=Search"},
                    "after": {"app_context": "browser", "focused": "search"},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.world_model.record_transition(
            {"app_context": "browser"},
            UiAction("click", "name=Search"),
            {"app_context": "browser", "focused": "search"},
        )
        return {"transitions": transitions, "archives_parsed": len(paths)}


class AdaptationTrainingTests(unittest.TestCase):
    def test_fetch_json_retries_transient_http_error(self) -> None:
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b'{"rows": []}'

        transient_error = urllib.error.HTTPError(
            url="https://example.test/rows",
            code=502,
            msg="Bad Gateway",
            hdrs=Message(),
            fp=io.BytesIO(b""),
        )

        with (
            mock.patch("agentos_orchestrator.cognition.adaptation_training.time.sleep"),
            mock.patch(
                "agentos_orchestrator.cognition.adaptation_training.urllib.request.urlopen",
                side_effect=[transient_error, _Response()],
            ) as mocked_urlopen,
        ):
            payload = StubAdaptationTrainer.fetch_json("https://example.test/rows")

        transient_error.close()

        self.assertEqual(payload, {"rows": []})
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_trainer_ingests_screenspot_rows_and_persists_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            buf = io.BytesIO()
            Image.new("RGB", (400, 200), color=(255, 255, 255)).save(buf, format="PNG")
            trainer = StubAdaptationTrainer(temp_dir, buf.getvalue())

            result = trainer.train(
                AdaptationTrainingConfig(
                    screenspot_limit=1,
                    include_internal_trajectories=False,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.screenspot_rows_used, 1)
            self.assertTrue(Path(result.local_vla_model_path).exists())
            self.assertTrue(Path(result.screenspot_rows_path).exists())
            self.assertTrue(Path(result.osworld_manifest_path).exists())
            self.assertIn("gimp", result.osworld_domains)

    def test_trainer_replays_internal_trajectories_into_world_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TrajectoryRecorder(temp_dir)
            path = recorder.start_run("adapt-run", "type in an unknown editor")
            assert path is not None
            recorder.record_step(
                run_id="adapt-run",
                objective="type in an unknown editor",
                option_name="fill_form",
                before=AbstractUIState(app_context="text_editor"),
                after=AbstractUIState(
                    app_context="text_editor",
                    elements=[],
                ),
                action=UiAction("set_text", "name=Editor", value="hello"),
                expected_observation="Text appears in the editor.",
                receipt=json.dumps({"status": "typed"}),
                outcome=OutcomeEvaluation(
                    expected="Text appears in the editor.",
                    observed="typed",
                    matched=True,
                ),
                capability_profile={"app_family": "editor"},
                adapter_context={"family": "editor"},
                verification_contract={"kind": "field_contains"},
                verification_result={"matched": True},
            )
            recorder.finish_run("adapt-run", {"success": True})

            buf = io.BytesIO()
            Image.new("RGB", (400, 200), color=(240, 240, 240)).save(buf, format="PNG")
            trainer = StubAdaptationTrainer(temp_dir, buf.getvalue())
            result = trainer.train(
                AdaptationTrainingConfig(
                    screenspot_limit=0,
                    include_internal_trajectories=True,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.world_model_transitions, 1)
            self.assertTrue(Path(result.world_model_checkpoint).exists())
            self.assertTrue(Path(result.internal_dataset_path).exists())

    def test_trainer_ingests_click100k_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            buf = io.BytesIO()
            Image.new("RGB", (400, 200), color=(250, 250, 250)).save(buf, format="PNG")
            trainer = StubAdaptationTrainer(temp_dir, buf.getvalue())

            result = trainer.train(
                AdaptationTrainingConfig(
                    screenspot_limit=0,
                    click100k_limit=1,
                    include_internal_trajectories=False,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.click100k_rows_used, 1)
            self.assertTrue(Path(result.click100k_rows_path).exists())

    def test_trainer_reports_underfill_and_scale_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            buf = io.BytesIO()
            Image.new("RGB", (400, 200), color=(250, 250, 250)).save(buf, format="PNG")
            trainer = StubAdaptationTrainer(temp_dir, buf.getvalue())

            result = trainer.train(
                AdaptationTrainingConfig(
                    screenspot_limit=0,
                    gui_actor_limit=5,
                    include_internal_trajectories=False,
                    download_osworld_manifest=False,
                )
            )

            self.assertFalse(result.success)
            self.assertTrue(result.underfill["underfilled"])
            self.assertEqual(
                result.underfill["items"]["gui_actor_rows"]["missing"],
                5,
            )
            self.assertFalse(result.scale_report["meets_minimum_scale"])
            self.assertIn("Scale target not yet met", " ".join(result.notes))

    def test_trainer_replays_osworld_archives_into_world_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            buf = io.BytesIO()
            Image.new("RGB", (400, 200), color=(240, 240, 240)).save(buf, format="PNG")
            trainer = StubAdaptationTrainer(temp_dir, buf.getvalue())

            result = trainer.train(
                AdaptationTrainingConfig(
                    screenspot_limit=0,
                    include_internal_trajectories=False,
                    osworld_archive_limit=1,
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.osworld_archive_transitions, 1)
            self.assertEqual(result.osworld_archives_parsed, 1)
            self.assertTrue(Path(result.osworld_archive_dataset_path).exists())

    def test_gui_actor_training_sample_resolves_archive_member_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            buf = io.BytesIO()
            Image.new("RGB", (320, 160), color=(230, 230, 230)).save(buf, format="PNG")
            trainer = StubAdaptationTrainer(temp_dir, buf.getvalue())
            archive_path = Path(temp_dir) / "gui_actor.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "AndroidControl/tfrecord/images/android_control_episode_[1]_0.png",
                    buf.getvalue(),
                )
            row = {
                "image": "android_control_episode_[1]_0.png",
                "conversations": [
                    {"from": "human", "value": "<image> Click search"},
                    {"from": "gpt", "bbox_gt": [0.1, 0.2, 0.3, 0.4]},
                ],
            }

            with zipfile.ZipFile(archive_path) as archive:
                sample = trainer.gui_actor_training_sample(
                    row,
                    archive,
                    "androidcontrol",
                )

            self.assertIsNotNone(sample)
            assert sample is not None
            self.assertEqual(
                sample[4]["archive_member"],
                "AndroidControl/tfrecord/images/android_control_episode_[1]_0.png",
            )

    def test_iter_transition_candidates_parses_osworld_traj_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            buf = io.BytesIO()
            Image.new("RGB", (320, 160), color=(220, 220, 220)).save(buf, format="PNG")
            trainer = StubAdaptationTrainer(temp_dir, buf.getvalue())

            payload = {
                "kind": "osworld_traj",
                "archive_member": "libreoffice_writer/task-1/traj.jsonl",
                "app_context": "libreoffice_writer",
                "records": [
                    {
                        "step_num": 1,
                        "action": "pyautogui.click(x=0.5, y=0.2)",
                        "response": "clicked",
                        "reward": 0,
                        "done": False,
                        "screenshot_file": "step_1.png",
                    },
                    {
                        "step_num": 2,
                        "action": "DONE",
                        "response": "finished",
                        "reward": 1,
                        "done": True,
                        "screenshot_file": "step_2.png",
                    },
                ],
            }

            transitions = list(trainer.iter_transition_candidates(payload))

            self.assertEqual(len(transitions), 2)
            before, action, after = transitions[0]
            self.assertEqual(before["app_context"], "libreoffice_writer")
            self.assertEqual(action.action_type, "click")
            self.assertEqual(after["step_num"], 2)

    def test_distributed_targets_balances_across_sources(self) -> None:
        self.assertEqual(_distributed_targets(16, 5), [4, 3, 3, 3, 3])
        self.assertEqual(_distributed_targets(3, 5), [1, 1, 1, 0, 0])
        self.assertEqual(_distributed_targets(0, 4), [0, 0, 0, 0])

    def test_osworld_batches_preserve_core_archives_with_overfetch(self) -> None:
        archives = [f"archive_{index:02d}.zip" for index in range(13)]

        batches = _osworld_shard_batches(
            archives,
            shard_count=6,
            archives_per_shard=3,
            candidate_multiplier=2,
        )

        self.assertEqual(
            batches,
            [
                ["archive_00.zip", "archive_01.zip", "archive_02.zip"],
                ["archive_03.zip", "archive_04.zip", "archive_05.zip"],
                ["archive_06.zip", "archive_07.zip", "archive_08.zip"],
                ["archive_09.zip", "archive_10.zip", "archive_11.zip"],
                ["archive_12.zip"],
                [],
            ],
        )

    def test_train_long_run_batches_archives_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            buf = io.BytesIO()
            Image.new("RGB", (400, 200), color=(240, 240, 240)).save(buf, format="PNG")
            trainer = StubAdaptationTrainer(temp_dir, buf.getvalue())
            state_path = Path(temp_dir) / "long_run_state.json"

            result = trainer.train_long_run(
                AdaptationLongRunConfig(
                    shard_count=2,
                    screenspot_limit_per_shard=1,
                    click100k_limit_per_shard=1,
                    gui_actor_limit_per_shard=0,
                    include_internal_trajectories_first_shard=False,
                    osworld_archives_per_shard=2,
                    osworld_archive_transition_limit_per_shard=4,
                    state_path=str(state_path),
                )
            )

            self.assertTrue(result.success)
            self.assertEqual(result.shards_completed, 2)
            self.assertEqual(result.total_shards, 2)
            self.assertEqual(result.screenspot_rows_used, 2)
            self.assertEqual(result.click100k_rows_used, 2)
            self.assertEqual(result.osworld_archives_parsed, 4)
            self.assertEqual(result.osworld_archive_transitions, 8)
            self.assertEqual(len(result.shard_results), 2)
            self.assertEqual(result.requested["osworld_archives"], 4)
            self.assertFalse(result.underfill["underfilled"])
            self.assertEqual(result.scale_report["minimum_target"], 100_000)
            self.assertTrue(state_path.exists())

            resumed = trainer.train_long_run(
                AdaptationLongRunConfig(
                    shard_count=2,
                    screenspot_limit_per_shard=1,
                    click100k_limit_per_shard=1,
                    gui_actor_limit_per_shard=0,
                    include_internal_trajectories_first_shard=False,
                    osworld_archives_per_shard=2,
                    osworld_archive_transition_limit_per_shard=4,
                    state_path=str(state_path),
                )
            )

            self.assertEqual(resumed.shards_completed, 2)
            self.assertEqual(len(resumed.shard_results), 2)
            self.assertEqual(resumed.osworld_archive_transitions, 8)


if __name__ == "__main__":
    unittest.main()
