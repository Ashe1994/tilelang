# MLA Decode（Ascend 950PR）算子开发交接文档

> 更新：2026-11-25 ｜ 分支 `ascend/mla-decode`（HEAD `f5bad79f`）｜ 测试 28/28 通过
> 英文完整版见仓库：`examples/ascend/mla_decode/HANDOVER.md`

## 一、项目一句话

在 Ascend 950PR 上用 tilelang 实现 DeepSeek MLA 解码注意力
`O = softmax((Q·KVᵀ + Q_pe·K_peᵀ)·scale)·KV`（H=128, D=512, PE=64, KV 最长 32768, fp16），
并在同数据上对齐 CANN 官方实现（FIA）。

## 二、当前成果（已验证）

| 场景 | 结果 |
|---|---|
| B=1, kv=8192 | native 1549.4 µs → paged **52.6 µs = 29.5x**，43.4 TFLOPS，rel 5.4e-4 |
| B=64, kv=8192 | **1.49–1.61 ms**（~98–102 TFLOPS） |
| B=64, kv=32768 | **5.31–5.96 ms**（~100–112 TFLOPS） |
| B=64 kv=1024…32768 自动分档 | 0.277 / 0.440 / 0.768 / 1.496 / 2.815 / 5.312 ms |
| 硬件利用率 | cube 约占每 block 墙钟 47–52%，约峰值 22–24% |
| 与 CANN FIA 对比 | FIA 快 **1.5–1.6x**（151 TFLOPS，35% 峰值） |

文件（`examples/ascend/mla_decode/`）：native 参考实现、paged 生产实现（两 tile
CV 重叠调度 + split-KV merge）、persistent 变体、bench/、README、VERIFICATION、HANDOVER。

## 三、换电脑开发：迁移清单

1. **代码**：`git clone git@github.com:Ashe1994/tilelang.git`，切 `ascend/mla-decode`；
   Windows 推送需 `set GIT_SSH=C:\Windows\System32\OpenSSH\ssh.exe`。
2. **环境**：CANN 9.3.0（`source $ASCEND_HOME_PATH/set_env.sh`）+ Python 3.11/3.12 +
   `torch 2.10.0+cpu` + `torch_npu 2.10.0` + 本仓库 `pip install -e .`（或用匹配的
   `tilelang-0.1.15+cpu.git*.whl`，DT 主机 `/home/z00946738/` 有 wheel，
   搭建脚本见 `mla-opt-exp/tools/dt_setup_env.sh`）。
3. **环境变量**：`export TILELANG_DEFAULT_TARGET=ascend`，并加载 CANN 的 `set_env.sh`。
4. **验证**：`python -m pytest test_mla_decode.py test_mla_decode_paged.py -q`（28 通过）
   再跑 `python bench/bench_paged.py`（应约 52–54 µs）。
5. **服务器**：同一 IP `123.60.114.238`，PR 端口 10118（别名 `pr`）、DT 端口 10236（别名 `dt`），
   root + 密钥；ssh 配置在旧机 `D:\Work\AI_WorkSpace\.ssh\config`，复制即可。
   - PR：`source /data/h00662772/mla_opt/scripts/env.sh`（**先 source 再 cd**），
     staging `/data/h00662772/mla_opt_exp/val`；
   - DT：`source /data/h00662772/env.sh`，staging `/data/h00662772/mla_opt_exp/dt_v1`。
   - 跑测/采样必须 `flock /dev/shm/mla_bench.lock`（PR）或 `/dev/shm/mla_dt.lock`（DT）。
6. **本地实验目录**：`D:\Work\AI_WorkSpace\mla-opt-exp\`（225 MB）建议整体拷走：
   `tools/`（remote.ps1、capture_msprof.sh、capture_insight.sh、实验生成脚本）、
   `opt/`（40+ 个历史探针）、`msprof/`（8 份 msprof 采集 + 4 份 Insight bin + 2 份带时间线采集）、
   `intake/`（提交用 commit message）。

## 四、复现命令

```bash
export TILELANG_DEFAULT_TARGET=ascend
cd examples/ascend/mla_decode
python -m pytest test_mla_decode.py test_mla_decode_paged.py -q
python bench/bench_paged.py                 # B=1 延迟 + 精度护栏
python bench/bench_persistent.py            # 多 batch / ragged 对比
python bench/bench_batch64.py --lens 8192 --batch 64

# 利用率采集
msprof op --warm-up=10 --launch-count=5 \
  --aic-metrics=PipeUtilization,Memory,MemoryL0,ResourceConflictRatio,L2Cache \
  --output=/dev/shm/prof/paged python bench/prof_paged.py --block-size 64
python bench/parse_pipes.py paged=/dev/shm/prof/paged

# MindStudio Insight（时间线必须带 PipeTimeline！）
msopprof --output=/dev/shm/insight/paged \
  --aic-metrics=Default,Roofline,Occupancy,BasicInfo,MemoryDetail,PipeUtilization,PipeTimeline \
  python bench/prof_multi.py --variant paged --batch 64 --lens 8192
# 导入 OPPROF_*/visualize_data.bin，并保留同级 dump/ 目录
```

## 五、经验教训（最重要的一节）

### 有效的设计

- **两 tile CV 重叠调度**（默认）：K/K_pe 双缓冲 + 两个 L0C 分块，slot1 的 QK 与 slot0
  的 softmax 重叠，phase1 47.7 → 43.6 µs（−8.6%）。
- **split-KV + merge**：把 KV 切成 ~32 个 tile 的 split，`batch*NS` 个 block 带来免费的
  跨 block CV 重叠；`pick_num_split` 自动分档（32 tiles/split、下限 2、上限 8、
  batch 8–16 下限 8），已对齐实测最优。
- **测量纪律**：`do_bench(event)` + `warm-up=10/repeat=50`，2–3 轮中位；msprof 用
  `PipeUtilization,Memory,MemoryL0,ResourceConflictRatio,L2Cache`；共享机高负载时重测。

### 硬约束（不要再撞）

- **L1 GEMM 只支持 NT**（`trans_A=False, trans_B=True`）→ V 必须以 Vᵀ 形式进 L1，
  KV 必须读两遍；FIA 也读两遍（1.19 GB），所以双读不是差距根源。
- **`T.dual_copy` 要求 M=128** → 不能用 64 行 head tile 去换 L1 空间做更深流水。
- **`if` 包裹 tile body 会破坏 ThreadSync**（barrier 被提出分支，race 仍在）→
  无法做 per-block 动态工作量（packed split plan 因此失败）。
- **`chunk_d=256`、单块 64 KiB Vᵀ、`block_N=128`、`num_split=32`、`TILE_SLOTS=3/4`、
  Vᵀ 双缓冲、融合 softmax、`simd_exp`、L2 提示、页表局部性、fp16 partial、persistent
  （均匀多 batch）、手写 mixed-kernel** —— 全部实测更差或无变化，详见
  `VERIFICATION.md` 第 8 节（含编译器报错原文）。
- PTO 方案已由决策排除；AscendC 上 `pv_l0` 会触发 AI core 故障 507015。

### 我们与 FIA 的差距

FIA（手写 AscendC）MTE2 占用 77.5%，我们约 59%；两者的 per-block DMA 速率接近，
差距是**流水重叠度**，不是带宽。可试的廉价手段都已被上面列出的实验封死，
要追平只能：①直接调用 FIA；②手写 AscendC kernel；③推动编译器支持
（NN GEMM / M<128 dual_copy / 可安全分支）。

## 六、后续建议（按性价比）

1. **A1：封装 CANN FIA 做解码注意力分发**（实测 1.5–1.6x）。入口按 shape/布局选择
   FIA（BNSD、`[num_blocks, N=1, block_size, D]`、fp16、`query_rope`/`key_rope`），
   我们的 kernel 作为回退，两路用同一 fp32 参考验证。
2. **服务层**：连续批处理（B=64 时每 token 成本比 B=1 好 2.1x）、按真实长度分桶编译、
   partial 缓冲预分配。
3. **手写 AscendC**（4 周量级）：同样的两 tile 算法 + 手动多缓冲/显式 flag + L0 NN 路径。
4. 小项：batch 16/32 的 split 表、merge 融合、FIA 的 PipeTimeline 采集做算子间对比。

## 七、常用入口

- 仓库文档：`README.md`（使用/性能）、`VERIFICATION.md`（自验证报告 + dead ends）、
  `HANDOVER.md`（英文完整交接）。
- 远程操作：`mla-opt-exp/tools/remote.ps1 -Action push|pull -HostAlias pr|dt ...`。
- 采样导出：`capture_msprof.sh`、`capture_insight.sh`；本地归档在
  `D:\Work\AI_WorkSpace\mla-opt-exp\msprof\`。
- DT 当前只保留验证树（240 KB，模块 md5 `b979177e…` 与仓库一致）；
  `tl_cache`、`/dev/shm` 内文件均为可再生成内容，可随时清理。
