# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler.record_function 接口功能正确性
API 名称：torch.autograd.profiler.record_function
API 签名：
  class record_function(_ContextDecorator):
      def __init__(self, name: str, args: str | None = None)

  关键方法：
      __enter__(self) -> record_function
      __exit__(self, exc_type, exc_value, traceback) -> None
      _call_end_callbacks_on_future(self, fut) -> Future

覆盖维度表：
| 覆盖维度         | 说明                                                                   | 覆盖情况                                                                  |
|------------------|------------------------------------------------------------------------|---------------------------------------------------------------------------|
| 空/非空          | name 必填非空；args 可为 None 或非空字符串                             | 已覆盖：test_name_stored / test_args_default_none / test_args_explicit    |
| 枚举选项         | 用法：上下文管理器 vs 函数装饰器                                       | 已覆盖：test_context_manager / test_decorator_usage                       |
| 参数类型         | name: str；args: str 或 None                                           | 已覆盖：test_missing_name_raises / test_args_explicit                     |
| 传参与不传参     | args 缺省 None vs 显式传入字符串                                       | 已覆盖：test_args_default_none / test_args_explicit                       |
| 等价类/边界值    | name 为空字符串；name 为长字符串；嵌套调用                             | 已覆盖：test_name_empty_string / test_name_long_string / test_nested      |
| 正常传参场景     | __enter__ 返回 self；退出后 record 有效；profiler 中标签可见           | 已覆盖：test_enter_returns_self / test_profiler_captures_label_npu        |
| 异常传参场景     | 不传 name 抛 TypeError；上下文中异常正常传播                           | 已覆盖：test_missing_name_raises / test_exception_propagates              |
| 硬件感知         | NPU 算子在 record_function 块内执行；标签出现在 profiler 事件中         | 已覆盖：test_profiler_captures_label_npu / test_nested_labels_npu         |
| 混合设备类型     | record_function 不接受 Tensor 输入，不涉及设备迁移                     | 未覆盖：API 为标注工具，无张量入参                                        |

未覆盖项及原因：
- 混合设备类型：record_function 是上下文标注工具，不操作张量，无设备混合路径。

注意：本测试仅验证功能正确性（调用不报错、标签可见性、状态正确性），
     不做精度和数值正确性校验。
"""

import inspect

import torch
import torch_npu  # noqa: F401

from torch_npu.testing.testcase import TestCase, run_tests
from torch.autograd.profiler import profile, record_function


class TestRecordFunction(TestCase):
    """Tests for torch.autograd.profiler.record_function."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------
    # API surface
    # ------------------------------------------------------------------

    def test_is_callable(self):
        """record_function is a class and instantiates without error."""
        self.assertTrue(callable(record_function))
        rf = record_function("label")
        self.assertIsInstance(rf, record_function)

    def test_name_stored(self):
        """name argument is stored on the instance."""
        rf = record_function("my_op")
        self.assertEqual(rf.name, "my_op")

    def test_args_default_none(self):
        """args defaults to None when not provided."""
        rf = record_function("label")
        self.assertIsNone(rf.args)

    def test_args_explicit(self):
        """args is stored when explicitly provided."""
        rf = record_function("label", args="extra_meta")
        self.assertEqual(rf.args, "extra_meta")

    def test_run_callbacks_on_exit_default(self):
        """run_callbacks_on_exit starts as True."""
        rf = record_function("label")
        self.assertTrue(rf.run_callbacks_on_exit)

    def test_missing_name_raises(self):
        """Omitting required name raises TypeError."""
        with self.assertRaises(TypeError):
            record_function()  # type: ignore[call-arg]

    def test_name_empty_string(self):
        """Empty string is a valid (if unusual) label."""
        rf = record_function("")
        self.assertEqual(rf.name, "")

    def test_name_long_string(self):
        """Long label strings are accepted."""
        long_name = "x" * 512
        rf = record_function(long_name)
        self.assertEqual(rf.name, long_name)

    def test_has_enter_exit(self):
        """record_function exposes __enter__ and __exit__."""
        self.assertTrue(hasattr(record_function, '__enter__'))
        self.assertTrue(hasattr(record_function, '__exit__'))

    # ------------------------------------------------------------------
    # Context manager behaviour
    # ------------------------------------------------------------------

    def test_context_manager(self):
        """record_function works as a with-statement context manager."""
        with record_function("cm_test"):
            x = torch.ones(4, device=self.device)
            _ = x.sum()

    def test_enter_returns_self(self):
        """__enter__ returns the record_function instance itself."""
        rf = record_function("enter_test")
        result = rf.__enter__()
        self.assertIs(result, rf)
        rf.__exit__(None, None, None)

    def test_context_manager_with_as(self):
        """'as' target in with statement is the instance."""
        with record_function("as_test") as rf:
            self.assertIsInstance(rf, record_function)

    def test_exception_propagates(self):
        """Exceptions inside the block propagate normally; context exits cleanly."""
        raised = False
        try:
            with record_function("exc_test"):
                raise ValueError("intentional")
        except ValueError:
            raised = True
        self.assertTrue(raised)

    def test_nested(self):
        """Nested record_function blocks do not interfere with each other."""
        with record_function("outer"):
            x = torch.ones(2, device=self.device)
            with record_function("inner"):
                _ = x + 1

    def test_multithreaded_interleaved(self):
        """Interleaved manual enter/exit does not raise (mirrors upstream test)."""
        rf_outer = record_function("outer_mt")
        rf_outer.__enter__()
        with record_function("inner_mt"):
            rf_outer.__exit__(None, None, None)
        # Second sequence: enter outer after inner was entered and exited
        rf_outer2 = record_function("outer_mt2")
        with record_function("inner_mt2"):
            rf_outer2.__enter__()
        rf_outer2.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Decorator behaviour
    # ------------------------------------------------------------------

    def test_decorator_usage(self):
        """record_function works as a function decorator."""

        @record_function("decorated_fn")
        def my_fn(x):
            return x * 2

        x = torch.ones(4, device=self.device)
        result = my_fn(x)
        self.assertEqual(result.shape, torch.Size([4]))
        self.assertEqual(result.device.type, self.device_name)

    def test_decorator_preserves_return(self):
        """Decorated function returns the expected value."""

        @record_function("ret_fn")
        def compute(x):
            return x.sum()

        x = torch.full((3,), 2.0, device=self.device)
        result = compute(x)
        self.assertEqual(result.shape, torch.Size([]))
        self.assertEqual(result.dtype, torch.float32)

    # ------------------------------------------------------------------
    # Profiler integration on NPU
    # ------------------------------------------------------------------

    def test_profiler_captures_label_npu(self):
        """Label appears in profiler function_events when NPU ops run inside block."""
        x = torch.randn(8, 8, device=self.device)
        with profile() as p:
            with record_function("npu_label_test"):
                _ = x.sum()
        events = p.function_events
        found = any("npu_label_test" in e.name for e in events)
        self.assertTrue(found,
                        f"Label 'npu_label_test' not found in events: "
                        f"{[e.name for e in events]}")

    def test_profiler_label_count(self):
        """Each record_function block appears exactly once per invocation."""
        x = torch.ones(4, device=self.device)
        with profile() as p:
            with record_function("count_once"):
                _ = x + 1
        events = p.function_events
        count = sum(1 for e in events if e.name == "count_once")
        self.assertEqual(count, 1)

    def test_nested_labels_npu(self):
        """Both outer and inner labels appear in profiler events."""
        x = torch.randn(4, device=self.device)
        with profile() as p:
            with record_function("outer_label"):
                _ = x * 2
                with record_function("inner_label"):
                    _ = x + 1
        events = p.function_events
        names = [e.name for e in events]
        self.assertTrue(any("outer_label" in n for n in names),
                        f"'outer_label' not found in {names}")
        self.assertTrue(any("inner_label" in n for n in names),
                        f"'inner_label' not found in {names}")

    def test_decorator_label_visible_in_profiler(self):
        """Decorator usage: label appears in profiler events."""

        @record_function("prof_decorator_fn")
        def workload(x):
            return x.relu()

        x = torch.randn(4, device=self.device)
        with profile() as p:
            workload(x)
        events = p.function_events
        found = any("prof_decorator_fn" in e.name for e in events)
        self.assertTrue(found,
                        f"Decorator label not found; events: "
                        f"{[e.name for e in events]}")

    def test_args_metadata_does_not_raise(self):
        """Passing args= metadata string is accepted and does not raise during profiling."""
        x = torch.ones(2, device=self.device)
        with profile():
            with record_function("op_with_args", args='{"shape": [2]}'):
                _ = x + 1

    def test_no_profiler_context_does_not_raise(self):
        """record_function outside any profiler context runs silently."""
        x = torch.randn(4, device=self.device)
        with record_function("no_profiler"):
            _ = x.abs()


if __name__ == "__main__":
    run_tests()
