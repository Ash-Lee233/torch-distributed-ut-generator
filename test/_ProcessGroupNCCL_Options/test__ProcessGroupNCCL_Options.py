# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.ProcessGroupNCCL.Options 接口功能正确性
API 名称：torch.distributed.ProcessGroupNCCL.Options
API 签名：
  class ProcessGroupNCCL.Options(Backend.Options):
      config: ProcessGroupNCCL.NCCLConfig
      is_high_priority_stream: bool
      split_from: ProcessGroupNCCL
      split_color: int

      def __init__(self, is_high_priority_stream: bool = False) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 默认构造 vs 显式 is_high_priority_stream                     | 已覆盖：test_default_construction / test_high_priority_true |
| 枚举选项         | is_high_priority_stream 取 True / False                      | 已覆盖：test_high_priority_true / test_high_priority_false |
| 参数类型         | is_high_priority_stream: bool                                | 已覆盖：test_is_high_priority_stream_field_type |
| 传参与不传参     | 缺省 vs 关键字传入                                            | 已覆盖：test_default_construction / test_high_priority_true |
| 等价类/边界值    | split_color 默认值；config 字段可读                          | 已覆盖：test_split_color_default / test_config_field |
| 正常传参场景     | 字段值与输入一致                                              | 已覆盖：test_high_priority_true                |
| 异常传参场景     | 缺失参数为可选；多余位置参数 -> TypeError                     | 已覆盖：test_extra_positional_raises           |
| 子类继承关系     | Options is subclass of Backend.Options                        | 已覆盖：test_subclass_of_backend_options       |
| 混合设备类型     | 该 API 为后端配置类，不接受 Tensor                            | 未覆盖：API 无 Tensor 输入                     |

未覆盖项及原因：
- 混合设备类型：ProcessGroupNCCL.Options 是后端配置类，不接受 Tensor。
- 当前 NPU 环境未编译 NCCL，ProcessGroupNCCL 不可用，主用例通过 import 守卫识别该情况，
  在 setUp 中根据 is_nccl_available() 决定是否跳过。

注意：torch_npu 环境通常不编译 NCCL；若 ProcessGroupNCCL 不可用，本测试用 SKIP
     方式记录该状态而非硬失败（依赖 unittest.skipUnless）。
本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import unittest

import torch
import torch.distributed as dist

import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests


_NCCL_AVAILABLE = (
    hasattr(dist, "is_nccl_available")
    and dist.is_nccl_available()
    and hasattr(dist, "ProcessGroupNCCL")
)


@unittest.skipUnless(
    _NCCL_AVAILABLE,
    "ProcessGroupNCCL is not available in this build "
    "(typical for torch_npu environments)."
)
class TestProcessGroupNCCLOptions(TestCase):
    """Tests for torch.distributed.ProcessGroupNCCL.Options."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.Options = dist.ProcessGroupNCCL.Options

    def test_default_construction(self):
        """Default construction sets is_high_priority_stream to False."""
        opts = self.Options()
        self.assertFalse(opts.is_high_priority_stream)

    def test_high_priority_true(self):
        """Construct with is_high_priority_stream=True."""
        opts = self.Options(is_high_priority_stream=True)
        self.assertTrue(opts.is_high_priority_stream)

    def test_high_priority_false(self):
        """Construct with is_high_priority_stream=False explicitly."""
        opts = self.Options(is_high_priority_stream=False)
        self.assertFalse(opts.is_high_priority_stream)

    def test_is_high_priority_stream_field_type(self):
        """is_high_priority_stream is a bool."""
        opts = self.Options(is_high_priority_stream=True)
        self.assertIsInstance(opts.is_high_priority_stream, bool)

    def test_config_field(self):
        """Options exposes a config NCCLConfig instance."""
        opts = self.Options()
        cfg = opts.config
        self.assertIsNotNone(cfg)
        # Required fields per the type stub
        self.assertTrue(hasattr(cfg, "blocking"))
        self.assertTrue(hasattr(cfg, "cga_cluster_size"))
        self.assertTrue(hasattr(cfg, "min_ctas"))
        self.assertTrue(hasattr(cfg, "max_ctas"))

    def test_split_color_default(self):
        """split_color attribute is accessible (int)."""
        opts = self.Options()
        self.assertIsInstance(opts.split_color, int)

    def test_subclass_of_backend_options(self):
        """Options is a subclass of Backend.Options."""
        Backend = dist.Backend
        self.assertTrue(issubclass(self.Options, Backend.Options))

    def test_extra_positional_raises(self):
        """Passing too many positional args raises TypeError."""
        raised = False
        try:
            self.Options(True, "extra")  # type: ignore[call-arg]
        except TypeError:
            raised = True
        self.assertTrue(raised)


class TestProcessGroupNCCLOptionsAvailability(TestCase):
    """Always-on tests for the availability flag itself."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")

    def test_is_nccl_available_callable(self):
        """is_nccl_available() returns a bool."""
        self.assertTrue(callable(dist.is_nccl_available))
        self.assertIsInstance(dist.is_nccl_available(), bool)

    def test_consistency_with_attr(self):
        """If is_nccl_available is True, ProcessGroupNCCL must be exposed."""
        if dist.is_nccl_available():
            self.assertTrue(hasattr(dist, "ProcessGroupNCCL"))


if __name__ == "__main__":
    run_tests()
