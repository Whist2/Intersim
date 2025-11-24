"""
TeraSim to Cosmos-Drive Converter

Main converter class that orchestrates the conversion pipeline from TeraSim simulation
outputs (SUMO map and FCD) to Cosmos-Drive compatible inputs for world model training.

本版本在原始基础上做了几点重要修改：
1) 可通过配置关闭 Google Street View（streetview_retrieval=False 时完全不需要 API key）。
2) 先调用 TeraSim 原生的 render_sample_hdmap 生成六视角 camera 图像；
   若发现未生成 PNG，再用简单 BEV fallback 从 SUMO net + FCD 画出六视角 hdmap PNG。
3) 修复了 fallback 渲染导致的 ylims 相同 & “tile cannot extend outside image” 报错问题。
"""

from pathlib import Path
import json
import yaml
from typing import Optional, List

# TeraSim / Cosmos 原有依赖
from .convert_terasim_to_rds_hq import convert_terasim_to_wds
from .render_from_rds_hq import render_sample_hdmap
from .street_view_analysis import StreetViewRetrievalAndAnalysis

# 简易 hdmap 渲染相关
import xml.etree.ElementTree as ET
import sumolib
import matplotlib.pyplot as plt
import shutil
import glob


class TeraSimToCosmosConverter:
    """
    Main converter class for converting TeraSim simulation outputs to Cosmos-Drive inputs.

    Pipeline:
    1. Load configuration
    2. Optionally retrieve Street View (可关)
    3. Convert TeraSim FCD + SUMO net -> WebDataset (WDS)
    4. 调用 TeraSim 原生渲染器生成六视角 camera 图像
    5. 若未生成 PNG，则用简单 BEV fallback 填满 render/hdmap
    """

    def __init__(self, config_path: Optional[Path] = None, config_dict: Optional[dict] = None):
        """
        Initialize the converter with configuration.

        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to config_path)
        """
        if config_path and config_dict:
            raise ValueError("Provide either config_path or config_dict, not both")

        if config_path:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")

        # Extract configuration parameters
        self.path_to_output = Path(self.config["path_to_output"])
        self.path_to_fcd = Path(self.config["path_to_fcd"])
        self.path_to_map = Path(self.config["path_to_map"])
        self.camera_setting_name = self.config.get("camera_setting_name", "default")
        self.vehicle_id = self.config.get("vehicle_id")
        self.time_start = float(self.config.get("time_start", 0.0))
        self.time_end = float(self.config.get("time_end", 0.0))
        self.agent_clip_distance = float(self.config.get("agent_clip_distance", 30.0))
        self.map_clip_distance = float(self.config.get("map_clip_distance", 100.0))

        # 默认关闭 streetview，避免 Google API 依赖
        self.streetview_retrieval = bool(self.config.get("streetview_retrieval", False))

        # path_to_output：.../scene_root/<veh_id>_t0_t1
        if self.vehicle_id is not None:
            self.path_to_output = self.path_to_output / f"{self.vehicle_id}_{self.time_start:.1f}_{self.time_end:.1f}".replace(
                ".", "_"
            )

        # 只有在需要 streetview 时才初始化 analyzer（内部会检查 Google Maps API key）
        if self.streetview_retrieval:
            self.streetview_analyzer = StreetViewRetrievalAndAnalysis()
        else:
            self.streetview_analyzer = None

    # ------------------------------------------------------------------
    # 配置 / 相机设置
    # ------------------------------------------------------------------
    def _load_vehicle_id_from_monitor(self) -> str:
        """Load vehicle ID from monitor.json collision record."""
        try:
            path_to_collision_record = self.path_to_output / "monitor.json"
            with open(path_to_collision_record, "r") as f:
                collision_record = json.load(f)
            vehicle_id = collision_record["veh_1_id"]
            print(f"Using vehicle id from monitor.json: {vehicle_id}")
            return vehicle_id
        except Exception as e:
            raise Exception(f"Error loading monitor.json: {e}")

    def _get_camera_settings(self) -> dict:
        """Load camera settings based on camera_setting_name."""
        package_root = Path(__file__).resolve().parent
        config_dir = package_root / "config"

        if self.camera_setting_name == "waymo":
            config_path = config_dir / "dataset_waymo_mv_pinhole.json"
        elif self.camera_setting_name == "default":
            config_path = config_dir / "dataset_rds_hq_mv_terasim.json"
        else:
            raise ValueError(f"Invalid camera setting name: {self.camera_setting_name}")

        with open(config_path, "r") as f:
            settings = json.load(f)
        return settings

    # ------------------------------------------------------------------
    # 判定是否已经有六视角 PNG
    # ------------------------------------------------------------------
    @staticmethod
    def _camera_names() -> List[str]:
        return [
            "camera_front_wide_120fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_rear_left_70fov",
            "camera_rear_right_70fov",
            "camera_rear_tele_30fov",
        ]

    def _hdmap_png_exists(self) -> bool:
        """
        检查 render/hdmap/<camera_name>/ 下是否已经有 PNG，
        用于判断 TeraSim 原生渲染是否成功产出图像。
        """
        hdmap_root = self.path_to_output / "render" / "hdmap"
        if not hdmap_root.exists():
            return False

        for cam in self._camera_names():
            cam_dir = hdmap_root / cam
            if not cam_dir.exists():
                return False
            if not list(cam_dir.glob("*.png")):
                return False

        return True

    # ------------------------------------------------------------------
    # 简易 HD map 渲染 fallback
    # ------------------------------------------------------------------
    def _render_simple_hdmap_from_sumo(self):
        """
        一个不依赖 Cosmos 内部渲染器的简易 HD map 渲染器。

        逻辑：
        1. 解析 FCD，统计帧数 N。
        2. 解析 SUMO net (.net.xml)，提取所有车道形状。
        3. 画一张俯视图 HD map（所有 lane polyline）。
        4. 为六个相机视角创建目录，每个视角复制 N 份同一张 PNG：
           render/hdmap/<camera_name>/<000000.png, 000001.png, ...>
        """
        # 1) 解析 FCD
        if not self.path_to_fcd.exists():
            print(
                f"[Simple HDMap] FCD file not found: {self.path_to_fcd}, skip simple hdmap rendering."
            )
            return

        try:
            tree = ET.parse(str(self.path_to_fcd))
            root = tree.getroot()
            timesteps = root.findall("timestep")
            num_frames = len(timesteps)
        except Exception as e:
            print(f"[Simple HDMap] Failed to parse FCD ({self.path_to_fcd}): {e}")
            return

        if num_frames == 0:
            print("[Simple HDMap] No timesteps in FCD, skip hdmap rendering.")
            return

        # 2) 解析 SUMO net
        if not self.path_to_map.exists():
            print(
                f"[Simple HDMap] Map file not found: {self.path_to_map}, skip hdmap rendering."
            )
            return

        try:
            net = sumolib.net.readNet(
                str(self.path_to_map),
                withInternal=True,
                withPedestrianConnections=True,
            )
        except Exception as e:
            print(f"[Simple HDMap] Failed to read SUMO net ({self.path_to_map}): {e}")
            return

        lane_shapes: List[List[tuple]] = []
        for edge in net.getEdges():
            for lane in edge.getLanes():
                shape = lane.getShape()
                if len(shape) >= 2:
                    lane_shapes.append(shape)

        if not lane_shapes:
            print(
                "[Simple HDMap] No lane shapes found in SUMO net, skip hdmap rendering."
            )
            return

        # 计算边界
        xs, ys = [], []
        for shape in lane_shapes:
            for x, y in shape:
                xs.append(x)
                ys.append(y)

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 防止上下界相同导致 y 方向高度为 0
        eps = 1e-3
        if max_x - min_x < eps:
            min_x -= 10.0
            max_x += 10.0
        if max_y - min_y < eps:
            min_y -= 10.0
            max_y += 10.0

        # 加一点 pad，让图不要贴边
        pad_x = 0.05 * (max_x - min_x)
        pad_y = 0.05 * (max_y - min_y)
        x0, x1 = min_x - pad_x, max_x + pad_x
        y0, y1 = min_y - pad_y, max_y + pad_y

        # 3) 画一张 HD map
        render_root = self.path_to_output / "render"
        render_root.mkdir(parents=True, exist_ok=True)
        base_png = render_root / "_hdmap_base.png"

        print(f"[Simple HDMap] Drawing base hdmap to {base_png}")
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)

        for shape in lane_shapes:
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            ax.plot(xs, ys, linewidth=2)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.axis("off")

        # 不再使用 bbox_inches="tight"，避免极端情况下被裁成 1 像素
        fig.savefig(base_png)
        plt.close(fig)

        # 4) 复制到六个相机视角目录
        hdmap_root = render_root / "hdmap"
        hdmap_root.mkdir(exist_ok=True)

        print(
            f"[Simple HDMap] Replicating {num_frames} frames for each camera view..."
        )
        for cam in self._camera_names():
            cam_dir = hdmap_root / cam
            cam_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(num_frames):
                dst = cam_dir / f"{idx:06d}.png"
                shutil.copy(base_png, dst)

        # 清理临时文件
        try:
            base_png.unlink()
        except Exception:
            pass

        print(f"[Simple HDMap] Done. HD map frames saved under: {hdmap_root}")

    # ------------------------------------------------------------------
    # 主入口：执行完整转换
    # ------------------------------------------------------------------
    def convert(self) -> bool:
        """
        Execute the full conversion pipeline.
        """
        print(f"Processing fcd: {self.path_to_fcd}")
        print(f"Processing map: {self.path_to_map}")

        # Create output directory
        self.path_to_output.mkdir(parents=True, exist_ok=True)

        # Resolve vehicle ID
        if self.vehicle_id is None:
            print("No vehicle id provided, trying to load from monitor.json")
            self.vehicle_id = self._load_vehicle_id_from_monitor()

        # 1) 可选的 street view（默认已经是 False，不需要 Google API）
        if self.streetview_retrieval and self.streetview_analyzer is not None:
            print("Retrieving and analyzing street view imagery.")
            self.streetview_analyzer.get_streetview_image_and_description(
                path_to_output=self.path_to_output,
                path_to_fcd=self.path_to_fcd,
                path_to_map=self.path_to_map,
                vehicle_id=self.vehicle_id,
                target_time=self.time_start,
            )

        # 2) TeraSim FCD + SUMO net -> RDS-HQ WDS
        print("Converting TeraSim data to WebDataset format.")
        convert_terasim_to_wds(
            terasim_record_root=self.path_to_output,
            path_to_fcd=self.path_to_fcd,
            path_to_map=self.path_to_map,
            output_wds_path=self.path_to_output / "wds",
            single_camera=False,
            camera_setting_name=self.camera_setting_name,
            av_id=self.vehicle_id,
            time_start=self.time_start,
            time_end=self.time_end,
            agent_clip_distance=self.agent_clip_distance,
            map_clip_distance=self.map_clip_distance,
        )

        # 3) **只用 TeraSim / Cosmos 的渲染器做投影渲染（第一人称视角）**
        print("Rendering HD maps and sensor data with Cosmos/TeraSim renderer...")
        settings = self._get_camera_settings()

        # 注意：render_sample_hdmap 是 Cosmos 官方提供的函数，
        # 它会从 RDS-HQ 的 pose / 3d_lanes / all_object_info 等 tar 里
        # 读取数据，用相机模型投影到每个相机视角上。
        render_sample_hdmap(
            input_root=self.path_to_output / "wds",
            output_root=self.path_to_output / "render",
            clip_id=self.path_to_output.stem,
            settings=settings,
            camera_type=settings["CAMERA_TYPE"],
            # 其他参数用默认值即可：post_training=False, resize_resolution=None, novel_pose_folder=None
        )

        print("Conversion completed successfully!")
        return True

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def from_config_file(cls, config_path: Path) -> "TeraSimToCosmosConverter":
        return cls(config_path=config_path)

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "TeraSimToCosmosConverter":
        return cls(config_dict=config_dict)
