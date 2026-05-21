# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.distributed_c10d.rendezvous 接口功能正确性
API 名称：torch.distributed.distributed_c10d.rendezvous
API 签名：rendezvous(url: str, rank: int = -1, world_size: int = -1, **kwargs)
API 说明：torch.distributed.distributed_c10d.rendezvous 是
          torch.distributed.rendezvous（即 torch/distributed/rendezvous.py）的导入引用，
         两者指向同一个函数。

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | url 非空必填；rank/world_size 默认 -1                        | 已覆盖：test_callable / test_file_rendezvous_returns_tuple |
| 枚举选项         | url 协议 file:// 与无效协议                                  | 已覆盖：test_file_rendezvous_returns_tuple / test_invalid_scheme_raises |
| 参数类型         | url:str；rank:int；world_size:int                            | 已覆盖：test_invalid_url_type_raises / test_invalid_rank_type_raises |
| 传参与不传参     | 默认 rank=-1 vs 显式指定                                     | 已覆盖：test_file_rendezvous_returns_tuple     |
| 等价类/边界值    | world_size = 1 文件 rendezvous                               | 已覆盖：test_file_rendezvous_returns_tuple     |
| 正常传参场景     | 返回 (store, rank, world_size) 三元组                        | 已覆盖：test_file_rendezvous_returns_tuple     |
| 异常传参场景     | url 非 str；rank 非整数；scheme 未注册                       | 已覆盖：test_invalid_url_type_raises 等       |
| 重导出一致性     | distributed_c10d.rendezvous 与 torch.distributed 顶层入口签名一致 | 已覆盖：test_reexport_is_callable             |
| 混合设备类型     | rendezvous 仅返回 Store，不涉及 Tensor                        | 未覆盖：API 无 Tensor 输入                     |

未覆盖项及原因：
- 混合设备类型：rendezvous 不接受张量输入，仅返回 c10d.Store。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import inspect
import os
import tempfile
import uuid

import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests

from torch.distributed.distributed_c10d import rendezvous


class TestDistributedC10dRendezvous(TestCase):
    """Tests for torch.distributed.distributed_c10d.rendezvous."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        super().tearDown()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_callable(self):
        """rendezvous is a callable with the documented signature."""
        self.assertTrue(callable(rendezvous))
        sig = inspect.signature(rendezvous)
        params = list(sig.parameters.keys())
        self.assertIn("url", params)
        self.assertIn("rank", params)
        self.assertIn("world_size", params)

    def test_file_rendezvous_returns_tuple(self):
        """file:// rendezvous returns (Store, rank, world_size)."""
        path = os.path.join(self._tmpdir, f"rdv_{uuid.uuid4().hex}")
        url = f"file://{path}?rank=0&world_size=1"

        gen = rendezvous(url)
        store, ret_rank, ret_world_size = next(gen)
        self.assertEqual(ret_rank, 0)
        self.assertEqual(ret_world_size, 1)
        self.assertIsNotNone(store)

    def test_file_rendezvous_with_explicit_rank(self):
        """file:// rendezvous accepts rank/world_size as kwargs."""
        path = os.path.join(self._tmpdir, f"rdv_{uuid.uuid4().hex}")
        url = f"file://{path}"
        gen = rendezvous(url, rank=0, world_size=1)
        store, ret_rank, ret_world_size = next(gen)
        self.assertEqual(ret_rank, 0)
        self.assertEqual(ret_world_size, 1)
        self.assertIsNotNone(store)

    def test_invalid_url_type_raises(self):
        """url not str or bytes raises RuntimeError."""
        raised = False
        try:
            rendezvous(12345)  # type: ignore[arg-type]
        except RuntimeError:
            raised = True
        self.assertTrue(raised)

    def test_invalid_rank_type_raises(self):
        """rank not an integer raises RuntimeError."""
        raised = False
        try:
            rendezvous("file:///tmp/x", rank="0", world_size=1)  # type: ignore[arg-type]
        except RuntimeError:
            raised = True
        self.assertTrue(raised)

    def test_invalid_world_size_type_raises(self):
        """world_size not an integer raises RuntimeError."""
        raised = False
        try:
            rendezvous("file:///tmp/x", rank=0, world_size="1")  # type: ignore[arg-type]
        except RuntimeError:
            raised = True
        self.assertTrue(raised)

    def test_invalid_scheme_raises(self):
        """Unsupported URL scheme raises RuntimeError."""
        raised = False
        try:
            gen = rendezvous("ftp://nowhere?rank=0&world_size=1")
            next(gen)
        except RuntimeError:
            raised = True
        self.assertTrue(raised)

    def test_reexport_is_callable(self):
        """torch.distributed.rendezvous resolves to a callable with same surface."""
        from torch.distributed import rendezvous as TopLevel
        # On some installs `torch.distributed.rendezvous` resolves to the submodule
        # rather than the function. Either way the function must be reachable.
        fn = TopLevel if callable(TopLevel) else getattr(TopLevel, "rendezvous")
        self.assertTrue(callable(fn))
        sig_a = inspect.signature(rendezvous)
        sig_b = inspect.signature(fn)
        self.assertEqual(list(sig_a.parameters.keys()), list(sig_b.parameters.keys()))


if __name__ == "__main__":
    run_tests()
