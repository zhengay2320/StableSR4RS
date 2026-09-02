# 批量解混：单一深度学习张量输出

将 `batch_unmixing_tensor.py` 放到五个端元提取脚本、`spectral_unmixing_pipeline.py` 和 `unmixing_core.py` 同一目录，直接运行：

```bash
python batch_unmixing_tensor.py
```

默认输入：

```text
E:\开源数据集\word_star\new_star\val\lr
```

默认输出：

```text
E:\开源数据集\word_star\new_star\val\lr_unmixing
```

每个输入 TIFF 最终只保留一个同名 `.npy`，例如：

```text
Landcover-118405.tiff -> Landcover-118405.npy
```

默认张量为 `float32 [10,H,W]`：

```text
0 vegetation abundance
1 water abundance
2 bare abundance
3 snow abundance
4 building abundance
5 vegetation uncertainty
6 water uncertainty
7 bare uncertainty
8 snow uncertainty
9 building uncertainty
```

其中 uncertainty 是近优解集合中对应类别丰度的 `max-min`，反映解混模型选择敏感性，不是概率。

为了让张量可以直接进入深度学习，最终结果不含 NaN：

- 正常估计：保留 `[abundance, uncertainty]`；
- 端元缺失、拟合失败、NoData 或无效位置：`abundance=0, uncertainty=1`。

因此：

```text
(0, 低 uncertainty) -> 有把握认为该类贡献接近0
(0, 1)               -> 无法估计，不应解释为类别不存在
```

中间端元提取结果和解混诊断文件全部写到临时目录，生成 `.npy` 后自动删除。

如果使用 TensorFlow/Keras 更喜欢 HWC，可将脚本顶部：

```python
OUTPUT_LAYOUT = "HWC"
```

此时输出形状为 `[H,W,10]`。
