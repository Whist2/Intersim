"""
simple_usage_example.py

示例：
1. 直接从 InterHub 缓存生成 RDS-HQ；
2. 使用鸟瞰图（BEV）GUI 手动点击选择 ego 车辆，再生成对应 RDS-HQ。

运行前请确认：
- InterHub 统一缓存已生成（如 data/1_unified_cache）；
- interhub_terasim_bridge.py 在同一 Python 包中并已按最新版本替换。
"""

from pathlib import Path

from interhub_terasim_bridge import (
    convert_interhub_scene_to_rds_hq,
    interactive_convert_interhub_scene_to_rds_hq,
)


def main():
    # ===== 基本配置 =====
    CACHE_PATH = "data/1_unified_cache"      # InterHub 统一缓存路径
    DATASET_NAME = "interaction_multi"       # 根据你统一的数据集命名调整
    OUTPUT_DIR = Path("outputs/example")     # 输出目录

    SCENE_IDX = 0        # 要转换的场景索引（0 ~ num_scenes-1）
    TIME_START = 0.0     # RDS-HQ 片段起始时间（秒）
    TIME_END = 10.0      # RDS-HQ 片段结束时间（秒）

    # 是否使用交互式选车（True：先弹出 BEV 界面手动选 ego；False：自动选 ego）
    USE_INTERACTIVE = True

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if USE_INTERACTIVE:
        print(">>> Running INTERACTIVE example: BEV GUI + manual ego selection...")
        rds_dir = interactive_convert_interhub_scene_to_rds_hq(
            cache_path=CACHE_PATH,
            scene_idx=SCENE_IDX,
            time_start=TIME_START,
            time_end=TIME_END,
            output_dir=OUTPUT_DIR,
            dataset_name=DATASET_NAME,
            snapshot_time=None,          # None 表示在 [time_start, time_end] 的中点生成 BEV
            streetview_retrieval=False,  # 如需街景检索可改为 True（取决于 TeraSim 侧实现）
            agent_clip_distance=80.0,
            map_clip_distance=100.0,
            camera_setting="default",    # 或 "waymo" / 其他自定义设置
        )
    else:
        print(">>> Running NON-INTERACTIVE example: auto-select ego...")
        rds_dir = convert_interhub_scene_to_rds_hq(
            cache_path=CACHE_PATH,
            scene_idx=SCENE_IDX,
            time_start=TIME_START,
            time_end=TIME_END,
            output_dir=OUTPUT_DIR,
            dataset_name=DATASET_NAME,
            ego_agent_id=None,           # None 时由桥接代码自动选择 ego
            streetview_retrieval=False,
            agent_clip_distance=80.0,
            camera_setting="default",
        )

    print("\n=== Done ===")
    print("RDS-HQ output directory:")
    print(rds_dir)


if __name__ == "__main__":
    main()
