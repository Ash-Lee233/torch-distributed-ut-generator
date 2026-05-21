# UT 执行报告：`torch.distributed.ProcessGroupNCCL.Options`

**生成时间**：2026-05-21 11:45:19

## 执行命令

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5
python -m pytest test/<dir>/test_*.py -v --tb=short --no-header -q
```

## 环境摘要

- Python 3.11.14
- torch= 2.7.1+cpu
- torch_npu= 2.7.1.post2
- npu_device_count= 8
- npu_available= True
- CANN home: /usr/local/Ascend/cann-8.5.1


## 统计

- 通过: 2
- 跳过: 8
- 失败: 0
- 报错: 0
- 耗时: 10.4s
- pytest 概要: `======================== 2 passed, 8 skipped in 10.39s =========================`

## 测试结果

| 测试方法 | 结果 |
|----------|------|
| `TestProcessGroupNCCLOptions::test_config_field` | SKIPPED |
| `TestProcessGroupNCCLOptions::test_default_construction` | SKIPPED |
| `TestProcessGroupNCCLOptions::test_extra_positional_raises` | SKIPPED |
| `TestProcessGroupNCCLOptions::test_high_priority_false` | SKIPPED |
| `TestProcessGroupNCCLOptions::test_high_priority_true` | SKIPPED |
| `TestProcessGroupNCCLOptions::test_is_high_priority_stream_field_type` | SKIPPED |
| `TestProcessGroupNCCLOptions::test_split_color_default` | SKIPPED |
| `TestProcessGroupNCCLOptions::test_subclass_of_backend_options` | SKIPPED |
| `TestProcessGroupNCCLOptionsAvailability::test_consistency_with_attr` | PASSED |
| `TestProcessGroupNCCLOptionsAvailability::test_is_nccl_available_callable` | PASSED |

## 跳过用例分析

| 测试方法 | 跳过条件 | 跳过原因 | 合理性评估 |
|----------|----------|----------|------------|
| `TestProcessGroupNCCLOptions::test_config_field` | is_nccl_available() == False | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：在 NPU 环境下 NCCL 未编译属于预期 |
| `TestProcessGroupNCCLOptions::test_default_construction` | is_nccl_available() == False | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：在 NPU 环境下 NCCL 未编译属于预期 |
| `TestProcessGroupNCCLOptions::test_extra_positional_raises` | is_nccl_available() == False | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：在 NPU 环境下 NCCL 未编译属于预期 |
| `TestProcessGroupNCCLOptions::test_high_priority_false` | is_nccl_available() == False | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：在 NPU 环境下 NCCL 未编译属于预期 |
| `TestProcessGroupNCCLOptions::test_high_priority_true` | is_nccl_available() == False | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：在 NPU 环境下 NCCL 未编译属于预期 |
| `TestProcessGroupNCCLOptions::test_is_high_priority_stream_field_type` | is_nccl_available() == False | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：在 NPU 环境下 NCCL 未编译属于预期 |
| `TestProcessGroupNCCLOptions::test_split_color_default` | is_nccl_available() == False | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：在 NPU 环境下 NCCL 未编译属于预期 |
| `TestProcessGroupNCCLOptions::test_subclass_of_backend_options` | is_nccl_available() == False | torch_npu 环境无 NCCL，ProcessGroupNCCL 不可用 | 合理：在 NPU 环境下 NCCL 未编译属于预期 |

## 本次改动文件

- `test/_ProcessGroupNCCL_Options/test_ProcessGroupNCCL_Options.py` 等同目录文件