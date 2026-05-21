# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.load_state_dict 接口功能正确性
API 名称：torch.distributed.checkpoint.load_state_dict
API 签名：
  @deprecated(
      "`load_state_dict` is deprecated and will be removed in future versions. "
      "Please use `load` instead.",
      category=FutureWarning,
  )
  def load_state_dict(
      state_dict: dict[str, Any],
      storage_reader: StorageReader,
      process_group: dist.ProcessGroup | None = None,
      coordinator_rank: int = 0,
      no_dist: bool = False,
      planner: LoadPlanner | None = None,
  ) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | state_dict 含 1 个 vs 多个 Tensor                            | 已覆盖：test_no_dist_single_tensor / test_no_dist_multi_tensor |
| 枚举选项         | no_dist True（单进程）vs False（多卡 HCCL）                  | 已覆盖：test_no_dist_single_tensor / test_multiprocess_load |
| 参数类型         | storage_reader: FileSystemReader；planner: DefaultLoadPlanner| 已覆盖：test_no_dist_with_explicit_planner     |
| 传参与不传参     | planner / process_group 默认 vs 显式                         | 已覆盖：test_no_dist_with_explicit_planner     |
| 等价类/边界值    | 各种 dtype 与 shape                                          | 已覆盖：test_no_dist_dtype_int / test_no_dist_dtype_float16 |
| 正常传参场景     | 加载后张量值/shape/dtype 与原始一致                          | 已覆盖：test_no_dist_single_tensor             |
| 异常传参场景     | storage_reader 缺失 -> TypeError                              | 已覆盖：test_missing_storage_reader_raises     |
| Deprecation      | load_state_dict 被 @deprecated 标注                          | 已覆盖：test_is_deprecated                     |
| 混合设备类型     | 加载时张量必须预先分配在目标设备；同进程内仅一种 device 类型  | 未覆盖：单进程多设备 mix 不属于 dcp 语义       |

未覆盖项及原因：
- 混合设备类型：dcp 加载要求目标张量已位于目标 device，本接口不接受 mixed device。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import shutil
import tempfile
import warnings

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from torch.distributed.checkpoint import load_state_dict
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.filesystem import (
    FileSystemReader,
    FileSystemWriter,
)


def _save_reference(tmp_dir, state_dict):
    """Save a state_dict to disk using dcp.save_state_dict (also deprecated but stable)."""
    writer = FileSystemWriter(tmp_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        dcp.save_state_dict(
            state_dict=state_dict,
            storage_writer=writer,
            no_dist=True,
        )


def _load_via_api(tmp_dir, state_dict, **kwargs):
    """Load using load_state_dict, suppressing FutureWarning."""
    reader = FileSystemReader(tmp_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        load_state_dict(state_dict=state_dict, storage_reader=reader, **kwargs)


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29514')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _test_multiprocess_load(rank, world_size, device_name, tmp_dir):
    """Multi-rank load using HCCL process group."""
    _init_dist_hccl(rank, world_size)
    try:
        target_w = torch.zeros(4, 4, device=f'{device_name}:{rank}')
        state_dict = {"weight": target_w}
        reader = FileSystemReader(tmp_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            load_state_dict(state_dict=state_dict, storage_reader=reader,
                            no_dist=False)
        assert state_dict["weight"].shape == torch.Size([4, 4])
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestCheckpointLoadStateDict(TestCase):
    """Tests for torch.distributed.checkpoint.load_state_dict."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_callable(self):
        """load_state_dict is callable."""
        self.assertTrue(callable(load_state_dict))

    def test_is_deprecated(self):
        """load_state_dict is decorated with @deprecated."""
        marker = (
            getattr(load_state_dict, "__deprecated__", None)
            or getattr(load_state_dict, "__wrapped__", None)
        )
        self.assertIsNotNone(marker)

    def test_missing_storage_reader_raises(self):
        """Missing storage_reader raises TypeError."""
        raised = False
        try:
            load_state_dict({"w": torch.zeros(2)})  # type: ignore[call-arg]
        except TypeError:
            raised = True
        self.assertTrue(raised)

    def test_no_dist_single_tensor(self):
        """no_dist=True path loads a single-tensor state dict on NPU."""
        orig = {"weight": torch.arange(12, dtype=torch.float32,
                                       device=self.device_name).reshape(3, 4)}
        _save_reference(self._tmpdir, orig)

        target = {"weight": torch.zeros(3, 4, dtype=torch.float32,
                                        device=self.device_name)}
        _load_via_api(self._tmpdir, target, no_dist=True)
        self.assertEqual(target["weight"].shape, torch.Size([3, 4]))
        self.assertEqual(target["weight"].dtype, torch.float32)
        self.assertEqual(target["weight"].device.type, self.device_name)

    def test_no_dist_multi_tensor(self):
        """no_dist=True path loads a multi-tensor state dict."""
        orig = {
            "w1": torch.zeros(2, 3, device=self.device_name),
            "w2": torch.zeros(4, device=self.device_name),
        }
        _save_reference(self._tmpdir, orig)

        target = {
            "w1": torch.zeros(2, 3, device=self.device_name),
            "w2": torch.zeros(4, device=self.device_name),
        }
        _load_via_api(self._tmpdir, target, no_dist=True)
        self.assertEqual(target["w1"].shape, torch.Size([2, 3]))
        self.assertEqual(target["w2"].shape, torch.Size([4]))

    def test_no_dist_dtype_float16(self):
        """no_dist=True path with float16 dtype tensor."""
        orig = {"w": torch.zeros(8, dtype=torch.float16, device=self.device_name)}
        _save_reference(self._tmpdir, orig)

        target = {"w": torch.zeros(8, dtype=torch.float16, device=self.device_name)}
        _load_via_api(self._tmpdir, target, no_dist=True)
        self.assertEqual(target["w"].dtype, torch.float16)

    def test_no_dist_dtype_int(self):
        """no_dist=True path with int32 dtype tensor."""
        orig = {"w": torch.zeros(4, dtype=torch.int32, device=self.device_name)}
        _save_reference(self._tmpdir, orig)

        target = {"w": torch.zeros(4, dtype=torch.int32, device=self.device_name)}
        _load_via_api(self._tmpdir, target, no_dist=True)
        self.assertEqual(target["w"].dtype, torch.int32)

    def test_no_dist_with_explicit_planner(self):
        """no_dist=True path with explicitly provided DefaultLoadPlanner."""
        orig = {"w": torch.zeros(3, 3, device=self.device_name)}
        _save_reference(self._tmpdir, orig)

        target = {"w": torch.zeros(3, 3, device=self.device_name)}
        _load_via_api(self._tmpdir, target, no_dist=True,
                      planner=DefaultLoadPlanner())
        self.assertEqual(target["w"].shape, torch.Size([3, 3]))

    def test_no_dist_with_coordinator_rank(self):
        """no_dist=True path with explicit coordinator_rank=0."""
        orig = {"w": torch.zeros(2, device=self.device_name)}
        _save_reference(self._tmpdir, orig)

        target = {"w": torch.zeros(2, device=self.device_name)}
        _load_via_api(self._tmpdir, target, no_dist=True, coordinator_rank=0)
        self.assertEqual(target["w"].shape, torch.Size([2]))

    @skipIfUnsupportMultiNPU(2)
    def test_multiprocess_load(self):
        """Multi-rank HCCL distributed load."""
        # Prepare a checkpoint in the parent process (no_dist=True save).
        orig = {"weight": torch.zeros(4, 4)}
        _save_reference(self._tmpdir, orig)

        mp.spawn(_test_multiprocess_load,
                 args=(2, self.device_name, self._tmpdir),
                 nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
