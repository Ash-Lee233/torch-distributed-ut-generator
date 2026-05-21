# UT 执行报告：`torch.distributed.distributed_c10d._coalescing_manager`

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

- 通过: 8
- 跳过: 0
- 失败: 0
- 报错: 0
- 耗时: 142.1s
- pytest 概要: `======================== 8 passed in 142.14s (0:02:22) =========================`

## 测试结果

| 测试方法 | 结果 |
|----------|------|

## 本次改动文件

- `test/_distributed_c10d__coalescing_manager/test_distributed_c10d__coalescing_manager.py` 等同目录文件