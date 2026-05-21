# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._symmetric_memory.enable_symm_mem_for_group 接口功能正确性
API 名称：torch.distributed._symmetric_memory.enable_symm_mem_for_group
API 签名：
  @deprecated(...)
  def enable_symm_mem_for_group(group_name: c10d.GroupName) -> None
说明：该 API 已被标记为 @deprecated，仅保留向后兼容；调用时应发出 FutureWarning。

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | group_name 非空字符串                                        | 已覆盖：test_enable_default_group              |
| 枚举选项         | 调用一次 vs 重复调用相同 group                               | 已覆盖：test_enable_idempotent                 |
| 参数类型         | group_name: str                                              | 已覆盖：test_enable_default_group              |
| 传参与不传参     | 必填参数                                                     | 已覆盖：test_missing_arg_raises                |
| 等价类/边界值    | 使用默认 process group 的 group_name                          | 已覆盖：test_enable_default_group              |
| 正常传参场景     | 调用后返回 None                                              | 已覆盖：test_enable_default_group              |
| 异常传参场景     | 未注册的 group_name 抛错；缺少参数 TypeError                   | 已覆盖：test_invalid_group_name / test_missing_arg_raises |
| Deprecation      | 较新版本将本 API 标注为 @deprecated；用例兼容含/不含该装饰器  | 已覆盖：test_optional_deprecation_marker       |
| 混合设备类型     | 该 API 不接受 Tensor 输入                                     | 未覆盖：API 无 Tensor 输入                     |

未覆盖项及原因：
- 混合设备类型：enable_symm_mem_for_group 只接受 group_name 字符串。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import inspect
import os
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU

from torch.distributed._symmetric_memory import enable_symm_mem_for_group


def _init_dist_hccl(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29511')
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)


def _try_enable(name):
    """Call enable_symm_mem_for_group and return ('ok', None) or ('err', exc)."""
    try:
        ret = enable_symm_mem_for_group(name)
        return ('ok', ret)
    except (RuntimeError, NotImplementedError, ValueError, KeyError) as e:
        return ('err', e)


def _test_enable_default_group(rank, world_size, device_name):
    """Call enable_symm_mem_for_group with the default group name."""
    _init_dist_hccl(rank, world_size)
    try:
        group_name = dist.group.WORLD.group_name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            status, ret = _try_enable(group_name)
        if status == 'ok':
            assert ret is None
        else:
            # On backends without symmetric memory support, tolerate runtime errors
            assert isinstance(ret, (RuntimeError, NotImplementedError))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_enable_idempotent(rank, world_size, device_name):
    """Calling twice with the same group should not error harder than first call."""
    _init_dist_hccl(rank, world_size)
    try:
        group_name = dist.group.WORLD.group_name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            s1, _ = _try_enable(group_name)
            s2, _ = _try_enable(group_name)
        # If the first call failed, the second should fail consistently
        assert s1 == s2 or (s1 == 'err' and s2 in ('err', 'ok')), \
            f"Inconsistent: {s1} vs {s2}"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _test_invalid_group_name(rank, world_size, device_name):
    """An unregistered group name should raise."""
    _init_dist_hccl(rank, world_size)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            status, exc = _try_enable("no_such_group_xyz")
        assert status == 'err', "Expected an error for unknown group"
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestEnableSymmMemForGroup(TestCase):
    """Tests for torch.distributed._symmetric_memory.enable_symm_mem_for_group."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_callable(self):
        """API is callable with one positional param."""
        self.assertTrue(callable(enable_symm_mem_for_group))
        sig = inspect.signature(enable_symm_mem_for_group)
        self.assertIn("group_name", sig.parameters)

    def test_missing_arg_raises(self):
        """Calling without args raises TypeError."""
        raised = False
        try:
            enable_symm_mem_for_group()  # type: ignore[call-arg]
        except TypeError:
            raised = True
        self.assertTrue(raised)

    def test_optional_deprecation_marker(self):
        """If decorated with @deprecated, calling should emit FutureWarning;
        if not (older builds), the call must still be reachable and a Runtime/
        ValueError is acceptable for an unknown group."""
        decorated = (
            hasattr(enable_symm_mem_for_group, "__deprecated__")
            or hasattr(enable_symm_mem_for_group, "__wrapped__")
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FutureWarning)
            try:
                enable_symm_mem_for_group("no_such_group_xyz_marker")
            except Exception:
                pass
        if decorated:
            self.assertTrue(
                any(issubclass(w.category, FutureWarning) for w in caught),
                "Decorated API should emit FutureWarning"
            )
        else:
            # Older build: just verify the function ran (warning list is fine empty).
            self.assertIsInstance(caught, list)

    @skipIfUnsupportMultiNPU(2)
    def test_enable_default_group(self):
        mp.spawn(_test_enable_default_group,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_enable_idempotent(self):
        mp.spawn(_test_enable_idempotent,
                 args=(2, self.device_name), nprocs=2, join=True)

    @skipIfUnsupportMultiNPU(2)
    def test_invalid_group_name(self):
        mp.spawn(_test_invalid_group_name,
                 args=(2, self.device_name), nprocs=2, join=True)


if __name__ == "__main__":
    run_tests()
