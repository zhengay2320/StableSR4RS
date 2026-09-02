from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance

# =========================
# 路径配置
# =========================

# HR
hr_dir = Path(
    "/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/train/GT_geo_rad_visual"
)

# LR
lr_dir = Path(
    "/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/train/LR"
)

# 退化表示
degraded_dir = Path(
    "/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/train/LR_bicubic"
)

# 输出文件夹
save_dir = Path(
    "/data/zhengay/EDiffSR-main/data/EDiffSR_worldstrat_rgb_x4_per_image/train/hr_lr"
)

save_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 残差显示增强倍数
# =========================
# 1.0 = 原始残差
# 4.0 = 残差亮度放大4倍，更容易观察
# 8.0 = 更明显
residual_scale = 4.0


# =========================
# 支持的图像格式
# =========================
image_extensions = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff",
    ".webp"
}


# =========================
# 获取图片
# =========================
def get_image_dict(folder):
    """
    根据文件名 stem 建立映射。

    例如：
        0001.png
        0001.jpg

    都对应：
        key = "0001"
    """
    image_dict = {}

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_dict[file_path.stem] = file_path

    return image_dict


hr_images = get_image_dict(hr_dir)
lr_images = get_image_dict(lr_dir)
degraded_images = get_image_dict(degraded_dir)


print("=" * 70)
print(f"HR 数量          : {len(hr_images)}")
print(f"LR 数量          : {len(lr_images)}")
print(f"LR_bicubic 数量  : {len(degraded_images)}")
print("=" * 70)


# =========================
# 找到三个文件夹共同存在的数据
# =========================
common_names = sorted(
    set(hr_images.keys())
    & set(lr_images.keys())
    & set(degraded_images.keys())
)

print(f"成功匹配的三元组数量: {len(common_names)}")


# =========================
# 检查缺失情况
# =========================
all_names = (
    set(hr_images.keys())
    | set(lr_images.keys())
    | set(degraded_images.keys())
)

missing_hr = all_names - set(hr_images.keys())
missing_lr = all_names - set(lr_images.keys())
missing_degraded = all_names - set(degraded_images.keys())

if missing_hr:
    print(f"\n缺少 HR 的数量: {len(missing_hr)}")
    print("前 10 个:", sorted(missing_hr)[:10])

if missing_lr:
    print(f"\n缺少 LR 的数量: {len(missing_lr)}")
    print("前 10 个:", sorted(missing_lr)[:10])

if missing_degraded:
    print(f"\n缺少 LR_bicubic 的数量: {len(missing_degraded)}")
    print("前 10 个:", sorted(missing_degraded)[:10])


# =========================
# 处理图片
# =========================
for i, name in enumerate(common_names):

    hr_path = hr_images[name]
    lr_path = lr_images[name]
    degraded_path = degraded_images[name]

    try:
        # =========================
        # 读取图片
        # =========================
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        degraded = Image.open(degraded_path).convert("RGB")

        # 原始尺寸
        hr_width, hr_height = hr.size
        lr_width, lr_height = lr.size
        deg_width, deg_height = degraded.size

        # =========================
        # LR、degraded 上采样到 HR 尺寸
        # =========================
        lr_up = lr.resize(
            (hr_width, hr_height),
            resample=Image.Resampling.LANCZOS
        )

        degraded_up = degraded.resize(
            (hr_width, hr_height),
            resample=Image.Resampling.LANCZOS
        )

        # =========================
        # 计算残差
        #
        # residual = |degraded - LR|
        #
        # 每个 RGB 通道分别做绝对差值
        # =========================
        residual = ImageChops.difference(
            degraded_up,
            lr_up
        )

        # =========================
        # 放大残差，方便肉眼观察
        # =========================
        if residual_scale != 1.0:
            residual_visual = ImageEnhance.Brightness(
                residual
            ).enhance(residual_scale)
        else:
            residual_visual = residual

        # =========================
        # 四张图横向拼接
        #
        # 1. residual
        # 2. degraded
        # 3. LR
        # 4. HR
        # =========================
        combined = Image.new(
            "RGB",
            (hr_width * 4, hr_height)
        )

        # 1. 残差
        combined.paste(
            residual_visual,
            (0, 0)
        )

        # 2. 退化图
        combined.paste(
            degraded_up,
            (hr_width, 0)
        )

        # 3. LR
        combined.paste(
            lr_up,
            (hr_width * 2, 0)
        )

        # 4. HR
        combined.paste(
            hr,
            (hr_width * 3, 0)
        )

        # =========================
        # 保存
        # =========================
        save_path = save_dir / f"{name}.png"

        combined.save(save_path)

        print(
            f"[{i + 1:5d}/{len(common_names):5d}] "
            f"{name} | "
            f"Residual x{residual_scale:g} | "
            f"Degraded: {deg_width}x{deg_height} -> "
            f"{hr_width}x{hr_height} | "
            f"LR: {lr_width}x{lr_height} -> "
            f"{hr_width}x{hr_height} | "
            f"HR: {hr_width}x{hr_height}"
        )

    except Exception as e:

        print("\n处理失败！")
        print(f"name     : {name}")
        print(f"degraded : {degraded_path}")
        print(f"LR       : {lr_path}")
        print(f"HR       : {hr_path}")
        print(f"错误信息 : {e}")


print("\n" + "=" * 70)
print("处理完成！")
print(f"共保存 {len(common_names)} 张对比图")
print(f"残差显示增强倍数: {residual_scale}")
print(f"保存目录: {save_dir}")
print("=" * 70)