# CuTeDSL CUDA driver 与 MLIR runtime 启动方式说明

CuTeDSL 生成的 GPU Kernel 有两条主要的调用路径：

- **MLIR runtime 路径（默认）**：在 Python 侧生成 host stub，并在 IR 中插入 `cuda.launch_ex`/`gpu.launch_func` 等调用，随后交由 MLIR Runtime 管理模块加载与 kernel 启动。
- **CUDA Driver 直连路径（`device_compilation_only=True`）**：只生成设备端 kernel 的 MLIR，直接把生成的 cubin 交给 CUDA Driver，通过 Python 的 driver API 加载模块并发射 kernel，不构建 host stub。

## MLIR runtime 路径

默认情况下 `device_compilation_only` 为 `False`，`generate_kernel_operands_and_types` 会把 Python 实参转换成 MLIR kernel 形参并作为 `kernel_operands` 插入到 launch 语句中，生成完整的 host + device IR。【F:python/CuTeDSL/cutlass/base_dsl/dsl.py†L1688-L1723】

内核体由 `_CutlassIrKernelGenHelper` 创建 `cuda.kernel` 函数，并通过 `CutlassBaseDSL.cuda_launch_func` 或 `gpu_launch_func` 在 host 侧生成对应的 launch 操作（包含 grid/block、shared memory、stream 等配置）。【F:python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py†L495-L590】【F:python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py†L636-L707】

## CUDA Driver 直连路径

当 DSL 初始化时把 `device_compilation_only` 设为 `True`，生成 kernel 时会跳过 host 端参数转换，`kernel_operands` 为空，表示不在 MLIR 中生成 host stub。【F:python/CuTeDSL/cutlass/base_dsl/dsl.py†L1697-L1707】【F:python/CuTeDSL/cutlass/base_dsl/dsl.py†L1814-L1828】

随后 `_execute_by_cuda_driver` 只构建包含 device kernel 的 `gpu.container_module`，运行 `generate_cubin` 生成 cubin，并直接通过 CUDA Driver API 完成模块加载与 `cuLaunchKernel` 式的调用，无需 MLIR runtime 参与。【F:python/CuTeDSL/cutlass/base_dsl/dsl.py†L1650-L1664】【F:python/CuTeDSL/cutlass/base_dsl/runtime/cuda.py†L401-L436】

## 何时选择哪条路径？

- **优先使用 MLIR runtime（默认）**：自动为 kernel 生成 host 入口、类型转换和 launch 配置，适合普通 Python 直接调用、需要 MLIR 运行时功能（如 async 依赖、PDL、cluster 配置）时使用。
- **选择 CUDA Driver 直连**：需要仅编译设备端、或想绕开 MLIR runtime 直接用 driver 控制加载/执行时启用 `device_compilation_only=True`。这种模式不会生成 host stub，需要调用方自行准备 grid/block/smem 以及参数指针，适合集成到自定义 runtime 中。
