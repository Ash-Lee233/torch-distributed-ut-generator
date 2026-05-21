# UT 执行报告：`torch.autograd.profiler.record_function`

**生成时间**：2026-05-21

## 执行命令

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
python -m pytest test/autograd_profiler_record_function/test_autograd_profiler_record_function.py -v --tb=short --no-header
```

## 环境摘要

- Python 3.11.14
- torch= 2.7.1+cpu
- torch_npu= 2.7.1.post2
- npu_device_count= 8
- npu_available= True
- CANN home: /usr/local/Ascend/cann-8.5.1

## 统计

- 通过: 23
- 跳过: 0
- 失败: 0
- 耗时: 14.76s

## 测试结果

| 测试方法 | 结果 |
|----------|------|
| `test_args_default_none` | PASS |
| `test_args_explicit` | PASS |
| `test_args_metadata_does_not_raise` | PASS |
| `test_context_manager` | PASS |
| `test_context_manager_with_as` | PASS |
| `test_decorator_label_visible_in_profiler` | PASS |
| `test_decorator_preserves_return` | PASS |
| `test_decorator_usage` | PASS |
| `test_enter_returns_self` | PASS |
| `test_exception_propagates` | PASS |
| `test_has_enter_exit` | PASS |
| `test_is_callable` | PASS |
| `test_missing_name_raises` | PASS |
| `test_multithreaded_interleaved` | PASS |
| `test_name_empty_string` | PASS |
| `test_name_long_string` | PASS |
| `test_name_stored` | PASS |
| `test_nested` | PASS |
| `test_nested_labels_npu` | PASS |
| `test_no_profiler_context_does_not_raise` | PASS |
| `test_profiler_captures_label_npu` | PASS |
| `test_profiler_label_count` | PASS |
| `test_run_callbacks_on_exit_default` | PASS |

## 跳过用例分析

无跳过用例。

## 本次改动文件

- `test/autograd_profiler_record_function/test_autograd_profiler_record_function.py`
- `test/autograd_profiler_record_function/UT_REPORT.md`
