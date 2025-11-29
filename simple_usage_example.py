"""
simple_usage_example.py

示例：
1. 交互模式：
   - 先通过 GUI 在本地 InterHub 场景中手动选择一个场景（带 GIF 预览）；
   - 再在该场景的鸟瞰图（BEV）中点击选择 ego 车辆；
   - 最后生成对应的 RDS-HQ（已按方案 A 自动收缩到 ego 存在的时间窗口）。

2. 非交互模式：
   - 直接指定 scene_idx，自动选择 ego，生成 RDS-HQ。

运行前请确认：
- InterHub 统一缓存已生成（如 data/1_unified_cache）；
- interhub_terasim_bridge.py 已替换为带交互 GUI / GIF / 路网 / 矩形车辆的版本。
"""

from pathlib import Path

from interhub_terasim_bridge import (
    convert_interhub_scene_to_rds_hq,
    interactive_convert_interhub_scene_to_rds_hq,
)


def main():
    # ===== 基本配置 =====
    CACHE_PATH = "data/1_unified_cache"      # InterHub 统一缓存路径
    DATASET_NAME = "interaction_multi"       # 根据你的数据集名称调整
    OUTPUT_DIR = Path("outputs/example")     # 输出目录

    # 时间窗口：用于 RDS-HQ 生成和预览 GIF
    TIME_START = 0.0
    TIME_END = 10.0

    # 是否使用交互模式
    # True  → 先 GUI 选场景（带 GIF 预览）→ 再 BEV 选车 → 生成 RDS-HQ
    # False → 直接使用下面的 SCENE_IDX 自动选 ego → 生成 RDS-HQ
    USE_INTERACTIVE = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if USE_INTERACTIVE:
        # 交互模式：scene_idx 设为 None，表示先用 GUI 从本地所有场景中选择一个
        print(">>> Running INTERACTIVE example:")
        print("    1) Scene selection GUI with GIF preview")
        print("    2) BEV GUI with manual ego selection")
        rds_dir = interactive_convert_interhub_scene_to_rds_hq(
            cache_path=CACHE_PATH,
            scene_idx=None,             # None → 先 GUI 选场景
            time_start=TIME_START,
            time_end=TIME_END,
            output_dir=OUTPUT_DIR,
            dataset_name=DATASET_NAME,
            snapshot_time=None,         # None → 在 [time_start, time_end] 中点生成 BEV
            streetview_retrieval=False, # 如需街景检索可改为 True（取决于 TeraSim 实现）
            agent_clip_distance=80.0,
            map_clip_distance=100.0,
            camera_setting="default",   # 或 "waymo" / 其他预设
        )
    else:
        # 非交互模式：直接指定一个场景索引，由代码自动选择 ego
        SCENE_IDX = 0   # 0 ~ num_scenes-1 之间，根据需要修改

        print(">>> Running NON-INTERACTIVE example:")
        print(f"    Fixed scene_idx = {SCENE_IDX}, auto-select ego...")
        rds_dir = convert_interhub_scene_to_rds_hq(
            cache_path=CACHE_PATH,
            scene_idx=SCENE_IDX,
            time_start=TIME_START,
            time_end=TIME_END,
            output_dir=OUTPUT_DIR / f"scene_{SCENE_IDX:04d}",
            dataset_name=DATASET_NAME,
            ego_agent_id=None,          # None → 自动按规则选择 ego
            streetview_retrieval=False,
            agent_clip_distance=80.0,
            camera_setting="default",
        )

    print("\n=== Done ===")
    print("RDS-HQ output directory:")
    print(rds_dir)


if __name__ == "__main__":
    main()
