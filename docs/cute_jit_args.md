# CuTeDSL JIT argument handling (torch.Tensor example)

- `BaseDSL._generate_jit_func_args_for_known_types` 只特判 `Constexpr` 等编译期常量，默认返回空列表，表示当前实参不是“已知类型”。
- `BaseDSL._generate_jit_func_args` 看到返回为空后，会从 `JitArgAdapterRegistry` 查找适配器；对 `torch.Tensor`，注册表里找到 `TensorAdapter`。
- `TensorAdapter` 使用 `from_dlpack` 把 Torch 张量包装成 `_Tensor` 并标记布局动态，提供 `__c_pointers__`/`__get_mlir_types__` 接口，让主机路径生成 memref 指针实参与 MLIR 类型列表。具体的 MLIR 类型由 `_Tensor.mlir_type` 调用 `_dltensor_wrapper.get_type(elem_mlir_type, assumed_align)` 生成；`DLTensorWrapper.get_type` 会先构造指向描述符的数据指针 (`cute.ptr`，含地址空间/对齐)，再与 DLPack 元数据合成 `cute.memref`。
- 因此传入 Torch 张量时，实际执行路径是：基类不处理 → 通过适配器转成 CuTe `_Tensor` → 读取 C 指针和 MLIR 类型（通过 DLTensorWrapper 映射为 memref 入口指针），生成 host stub 参数；算子重载发生在 `_Tensor`/`TensorSSA` 上。
