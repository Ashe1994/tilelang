# MLA Decode on Ascend 950 — 实现与踩坑笔记

> 移植上游 CUDA 版 `tilelang/examples/deepseek_mla/example_mla_decode.py` 的
> num_split=1 路径到 **Ascend 950PR**。
>
> 代码位置：**`tilelang/examples/ascend/`**
> 结果：**5 个用例在 `pto` 与 `ascend` 两个后端均通过**。

---

## 0. 环境与运行

| 项目 | 值 |
|---|---|
| 芯片 | Ascend950PR（NpuArch **3510** → 平台分支 `Ascend950`） |
| CANN | **9.3.0 必需**（9.1.0 缺 SIMD 内建，device 编译直接失败） |
| 仓库 | `tile-ai/tilelang`（服务器 `/data/h00662772/tilelang`，本地 `D:\Work\AI_WorkSpace\tilelang`） |
| 构建 | `USE_ASCEND=ON USE_CUDA=OFF pip install -v . --no-build-isolation` |
| 构建前提 | 系统 cmake 3.28（**不可用 conda 的 cmake 4.x**）；子模块 `3rdparty/{tvm,cutlass}` |
| 激活 | `source /data/h00662772/env-950.sh` |

```bash
source /data/h00662772/env-950.sh
cd /data/h00662772/tilelang/examples/ascend

TILELANG_DEFAULT_TARGET=pto    pytest test_mla_decode.py -q   # 5 passed
TILELANG_DEFAULT_TARGET=ascend pytest test_mla_decode.py -q   # 5 passed
```

交付：`example_mla_decode.py`（kernel）、`test_mla_decode.py`（golden + pytest）。
**代码中不出现 `target=`**，后端完全由环境变量选择。

已验证的提交：`876a59e`（当前 main）、`0bd33f9`（cannbot-skills 预置版本）——两份上均 5/5 通过。

---

## 1. 限制优化结果

| 原限制 | 现状 | 做法 |
|---|---|---|
| `dim` 固定 512 | ✅ **已解决**，任意 `dim % chunk_d == 0` | 列表推导 + `enumerate()` 静态展开（K1） |
| `kv_len` 必须整除 `block_N` | ✅ **已解决**，任意 `kv_len ≥ block_N` | 末块回退对齐 + 列掩码（K2） |
| 无流水 | ✅ **已解决** | `T.Pipelined(num_stages=2)`；版本数会撑爆 UB，故不加 `annotate_buffer_versions`（P10） |
| `heads` 必须是 128 的倍数 | ⚠️ **平台硬约束** | 见 K4 |

---

## 2. 踩过的坑

### P1. `use of undeclared identifier 'asc_pack_to_high'`

- **根因**：CANN 9.1.0 的 `x86_64-linux/asc/` 没有这批 SIMD 内建（`asc_pack_to_high` /
  `asc_pack_to_low` / `asc_storealign_pack_quarter`），9.3.0 才有。
- **解法**：换 CANN 9.3.0。
- **教训**：先核对平台支持矩阵。此前所有 `aicore timeout` / cube L1 error 都是此因。

### P2. `import tilelang` 命中旧版本

- **根因**：别的仓库 `set_env.sh` 把旧源码塞进 `PYTHONPATH`，遮蔽 site-packages。
- **解法**：独立 `env-950.sh`，不 source 其它仓库的环境脚本。

### P3. `shared.l0b capacity 65536 is insufficient ... 278528`

- **根因**：显式整块分配 L0A/L0B（`[128,512]` fp16 = 128 KiB），而 **L0B 只有 64 KiB**。
- **解法**：GEMM 用 **L1 操作数**，L0 交给编译器切分。

### P4. `PTO L0C->UB MTE source stride must be at least n_size`

- **根因**：**`T.dual_copy` 的 L0C→UB 要求源 M=128**。单变量实测：

  | block_H | block_N | dim | 结果 |
  |---|---|---|---|
  | **128** | 128 | 128 | ✅ |
  | **128** | 128 | **512** | ✅ |
  | 64 | 128 | 128 | ❌ |
  | 32 | 128 | 128 | ❌ |

- **解法**：`block_H` 固定 128（`ROWS = 64`）。

### P5. `Ascend L1 GEMM currently only supports trans_A=False, trans_B=True (NT)`

- **影响**：PV 的 B 必须是 `Vᵀ`。
- **解法**：`T.copy(V, v_l1, transpose=True)`（单位阵探针验证 maxdiff = 0.0）。

### P6. L1 缓冲的**动态行切片**不能作 GEMM 的 B 操作数

- **症状**：`T.gemm(..., v_l1[dc_i*DC:(dc_i+1)*DC, :], ...)` 编译通过但**结果全错**（maxdiff 1.55）。
- **解法**：每个输出列分块用**独立 L1 缓冲**。

### P7. `for i in range(n)` 是 TIR 循环，不能索引 Python 列表

- **症状**：`TypeError: list indices must be integers or slices, not Var`。
- **解法**：见 K1 —— 列表推导 + `enumerate()` 是 Python 层静态展开。

### P8. UB 装不下 `[64,512]` fp32 累加器

- **预算**：`o_ub` 128 KiB + `o_tmp_ub` 128 KiB ≈ 256 KiB > UB 可用约 216 KiB。
- **解法**：输出宽度按 `chunk_d` 分块；`o_ub` 保留 `[64,dim]` 作跨块累加器。

### P9. 两段 QK 各占一个 L0C 会顶到容量上限

- **解法**：先跑 pe 段 `clear_accum=True` → `dual_copy` 到 `s_pe_ub`；再跑 content 段
  `clear_accum=True` → `dual_copy` 到 `s_ub`，在 UB 相加。L0C 峰值降到 96 KiB。

### P10. `T.annotate_buffer_versions` 把 UB 撑爆

- **症状**：`Z3 loop scheduling failed: scope "shared" capacity 24064 is insufficient
  for minimum buffer usage 49536`。
- **根因**：给 `s_ub`/`s_pe_ub` 各加 2 版本（16 KiB × 2 × 2 = 64 KiB）超限。
- **解法**：去掉版本注解，只保留 `T.Pipelined`；仍获得明显加速
  （190 s vs 加流水前 375 s，同一构建对比）。

---

## 3. 实用技巧

### K1. Python 层静态展开（dim 通用化的关键）

```python
v_l1 = [T.alloc_l1((chunk_d, block_N), dtype) for _ in range(NUM_DC)]   # 列表推导分配
for i, vb in enumerate(v_l1):        # enumerate → Python 循环，i 是 int
    T.copy(KV[b, start:start+block_N, i*chunk_d:(i+1)*chunk_d], vb[0:chunk_d, 0:block_N], transpose=True)
```
对照：`for i in range(N)` 会被 TVM Script 解析成 `T.serial`，`i` 变成 `Var`。
**这是区分「静态展开」与「TIR 循环」的关键。**

### K2. 尾块用「回退对齐 + 列掩码」

```python
base  = kv * block_N
start = T.min(base, kv_len - block_N)   # 末块回退，保证读到的都是有效数据
lo    = base - start                    # 前一块已消费的列
hi    = kv_len - start                  # 有效列上界
...
s_ub[r, c] = T.if_then_else(c >= lo, T.if_then_else(c < hi, scaled, -inf), -inf)
```
好处：不越界读，也不会因残留脏数据乘 0 产生 NaN。

### K3. 归约用 SimtVF + reducer

```python
with T.SimtVF(threads=128):
    red = T.alloc_reducer((ROWS,), "float32", op="max")   # sum/max/min/bit*
    T.reducer_init(red)
    for r, c in T.Parallel(ROWS, block_N):
        T.reducer_update(red[r], s_ub[r, c])
    out = T.alloc_fragment((ROWS,), "float32")
    T.finalize_reducer(red, out)
```
行广播直接写 `m_ub[r]`，不需要 `T.tile.broadcast`（那是 910 的 tile API）。

### K4. 双 AIV 与 `heads` 约束

`T.dual_copy` 要求 M=128 → 每核 64 行。`heads < 128` 时末块只有部分有效行，
需要按 AIV 分别限长写回，而 `T.Kernel(N)` 形式下拿不到 `vid`/`sid`，
无法为两个 AIV 计算各自写回行数。故 `heads` 必须是 128 的倍数。

### K5. 调试三件套（定位 P4/P5/P6 靠这个）

1. **单位阵探针** `A = I` → 验证转置拷贝与 GEMM 语义（maxdiff=0 即正确）
2. **单块探针** `kv_len = block_N` → 区分"单块逻辑错"与"跨块累加错"
3. **中间结果导出** `dual_copy` 到 GM 与 torch 对齐 → 把范围从整个 kernel 收窄到 PV
   （`S maxdiff 2.6e-6`、`L maxdiff 1.1e-5`）

---

## 4. 最终结构

```
T.Kernel(batch * heads/128) as bx
  L1:  q(128,dim) | q_pe(128,pe) | k(64,dim) | k_pe(64,pe) | p(128,64)
       v[0..N-1] 各(chunk_d,64)                      —— 每块独立
  L0C: s(128,64) 32K | o(128,chunk_d) 64K            = 96K
  UB : s(64,64) | s_pe(64,64) | p(64,64)fp16
       o(64,dim) | o_tmp(64,chunk_d) | o_half(64,chunk_d)fp16

  for kv in T.Pipelined(KV_BLOCKS, num_stages=2):
      start = min(kv*block_N, kv_len-block_N)
      搬 KV/K_pe → L1；搬 KV 各列块 → V^T 的独立 L1 缓冲
      QK pe 段 gemm → dual_copy → s_pe_ub
      QK content 段 gemm → dual_copy → s_ub
      SimtVF: scale+列掩码 → reduce_max → alpha → exp → reduce_sum → l 更新 → P fp16
      dual_copy(p_ub → p_l1)
      for 每个列块: PV gemm → dual_copy → 累加到 o_ub 对应列块
  归一化 → 按列块转 fp16 → dual_copy 到 GM
```

---

## 5. 验证结果（提交 876a59e）

| 用例 | batch | heads | kv_len | dim | pe_dim | pto | ascend |
|---|---|---|---|---|---|---|---|
| tiny | 1 | 128 | 128 | 512 | 64 | ✅ | ✅ |
| kv_tail | 1 | 128 | **100** | 512 | 64 | ✅ | ✅ |
| dim256 | 1 | 128 | 128 | **256** | 64 | ✅ | ✅ |
| kv_multi_block | 1 | 128 | 512 | 512 | 64 | ✅ | ✅ |
| batch2_heads256 | 2 | 256 | 128 | 512 | 64 | ✅ | ✅ |

- 精度：`rtol=atol=1e-2`（fp16），比较前统一转 fp32
- 耗时：pto 全量 **481.6 s**；ascend 全量 **68.2 s**（同一构建；pto 首编较慢）

---

## 6. 后续可做

| 项 | 说明 |
|---|---|
| `heads` 灵活 | 需改 `MixedKernel` 拿 `sid`，才能为非 128 倍数做限长写回 |
| 版本数调优 | 逐个给 L1/UB 加 `annotate_buffer_versions`，每次核对 `scope capacity` |
| 性能剖析 | `do_bench(backend="msprof_detail")` 出 ms / TFLOPS |
| block_N 调参 | 当前 64；128 可减循环次数但 L1/UB 压力更大 |

---

## 7. 性能实测与调优

环境：Ascend950PR ×1（`ASCEND_RT_VISIBLE_DEVICES=6`），CANN 9.3.0，`TILELANG_DEFAULT_TARGET=pto`。
形状：`batch=1, heads=128, kv_len=8192, dim=512, pe_dim=64`（DeepSeek MLA decode 典型配置）。
FLOPs 口径：`2·b·h·s·(dim+pe) + 2·b·h·s·dim`。

### 7.1 参数扫描

| block_N | chunk_d | num_stages | threads | 编译 | 延迟 | TFLOP/s | 说明 |
|---|---|---|---|---|---|---|---|
| 64 | 128 | 2 | 128 | 101.5 s | 2.128 ms | 1.07 | 初始默认 |
| 64 | 128 | 3 | 128 | 87.8 s | 2.145 ms | 1.06 | 加深流水无收益 |
| 64 | 128 | 2 | **256** | 100.1 s | **1.801 ms** | 1.27 | +18% |
| 64 | 64 | 2 | 128 | 125.0 s | 2.215 ms | 1.03 | 更细分块更慢 |
| 64 | 128 | 3 | 256 | 91.5 s | 1.808 ms | 1.26 | 与 s2 持平 |
| 64 | 128 | 2 | **512** | 98.0 s | **1.597 ms** | **1.43** | **最优** |
| 64 | 128 | 3 | 512 | — | 1.601 ms | 1.42 | 持平 |
| 64 | 128 | 2 | 1024 | — | 1.630 ms | 1.40 | 回落 |
| 128 | 128 | 2 | 256 | — | 编译失败 | — | UB 超限 |
| 64 | 256 | 2 | 256 | — | 编译失败 | — | L0C/UB 超限 |

**最终默认：`block_N=64, chunk_d=128, num_stages=2, threads=512`**
（2.128 → 1.597 ms，约 **+25%**；正确性 maxdiff 1e-4，全量 5/5 通过）

### 7.2 为什么 block_N / chunk_d 不能再放大

`T.dual_copy` 要求 `block_H=128` → `ROWS=64`，输出累加器 `o_ub` 固定占
`64×dim×4B = 128 KiB`。剩余 UB 只够放 `s_ub`/`s_pe_ub`（`64×block_N×4B`）、
`p_ub`、`o_tmp_ub`/`o_half_ub`（按 `chunk_d`）。放大即报：

```
Z3 loop scheduling failed: scope "shared" capacity 89600 is insufficient
for minimum buffer usage 131584
```

L0C 同理：`s_l0c(128×64 fp32=32 KiB) + o_l0c(128×chunk_d×4B)`，`chunk_d=256` 即超 128 KiB。

### 7.3 真正的瓶颈：batch=1 只用一个核

| batch | 延迟 | TFLOP/s |
|---|---|---|
| 1 | 1.606 ms | 1.43 |
| 4 | 1.606 ms | 5.68 |
| 16 | 1.612 ms | **22.64** |

延迟几乎不随 batch 变化 → **并行度完全来自 batch**。
`block_H=128` 使 `grid = batch × (heads/128)`；当 `heads=128, batch=1` 时 **grid=1**，
只占用 1 个 AI Core，其余核空闲。

### 7.4 下一步：KV 切分（flash-decoding 风格）

把 `kv_len` 切成 `num_split` 份分给多个核，各自算局部 `O` 与 `LSE`，
再用第二个 `T.Kernel` 合并（上游 CUDA 版 `main_split` 即此结构）。
预期 batch=1 场景接近线性加速（上限为核数）。
代价：需要额外 GM scratch（`O_partial [num_split,b,h,dim]`、`LSE [num_split,b,h]`）
并增加一趟合并读写。
