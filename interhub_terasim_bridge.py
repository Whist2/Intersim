"""
InterHub-TeraSim Direct Integration Bridge (Cosmos-Drive-Dreams Enhanced Version)

功能概览：
1. 使用 InterHub 的统一轨迹数据生成完整 SUMO FCD 和路网；
2. 利用颜色配置，让生成的视频更符合 Cosmos-Drive-Dreams 的输入标准；
3. 新增：使用 InterHub 方法生成鸟瞰图（BEV），弹出 GUI 供用户手动点击选车（高亮显示），
   然后使用所选车辆作为 ego 生成对应的 RDS-HQ 视图。
"""

import os
import re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from tqdm import tqdm
import math
import imageio
from shapely.geometry import box as shapely_box, LineString
from matplotlib import patches
from trajdata import MapAPI
import math

# InterHub / trajdata dependencies
from trajdata.data_structures import Scene
from trajdata.caching import EnvCache
from trajdata.caching.df_cache import DataFrameCache
from trajdata.data_structures import AgentType

# TeraSim dependencies
from terasim_cosmos import TeraSimToCosmosConverter


class InterHubToTeraSimBridge:
    """
    Bridge between InterHub's unified trajectory data and TeraSim's RDS-HQ
    generation pipeline - Cosmos-Drive-Dreams ENHANCED VERSION.
    """

    # Vehicle dimension defaults by type
    VEHICLE_DIMENSIONS: Dict[AgentType, Dict[str, float]] = {
        AgentType.VEHICLE: {"length": 4.5, "width": 1.8, "height": 1.5},
        AgentType.PEDESTRIAN: {"length": 0.5, "width": 0.5, "height": 1.7},
        AgentType.BICYCLE: {"length": 1.8, "width": 0.6, "height": 1.7},
        AgentType.MOTORCYCLE: {"length": 2.0, "width": 0.8, "height": 1.5},
    }

    # 默认 Cosmos-Drive-Dreams 风格的 HDMap 颜色配置（RGB）
    DEFAULT_COSMOS_HD_MAP_COLOR_CONFIG: Dict[str, List[int]] = {
        # 背景（非道路）
        "background": [0, 0, 0],
        # 可行驶区域 / drivable area
        "drivable_area": [40, 40, 40],
        # 车道中心线 / lane center
        "lane_center": [255, 255, 255],
        # 车道边界 / road edge
        "road_edge": [255, 0, 0],
        # 人行横道
        "crosswalk": [255, 255, 0],
        # 停止线
        "stop_line": [255, 0, 255],
        # 路口区域
        "intersection": [0, 255, 255],
    }

    # HDMap 类型到语义 id 的映射
    DEFAULT_COSMOS_HD_MAP_TYPE_CONFIG: Dict[str, int] = {
        "drivable_area": 1,
        "road_edge": 2,
        "lane_center": 3,
        "crosswalk": 4,
        "stop_line": 5,
        "intersection": 6,
    }

    def __init__(
        self,
        interhub_cache_path: Union[str, Path],
        dataset_name: str = "interaction_multi",
        verbose: bool = True,
        scene_dt: float = 0.1,
        hdmap_color_config: Optional[Dict[str, List[int]]] = None,
        hdmap_type_config: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            interhub_cache_path: InterHub unified cache directory
            dataset_name: dataset/env name (e.g., "interaction_multi")
            verbose: whether to print progress
            scene_dt: scene time step in seconds (default 0.1)
            hdmap_color_config: optional HDMap color config (Cosmos style by default)
            hdmap_type_config: optional HDMap type/id config (Cosmos style by default)
        """
        self.cache_path = Path(interhub_cache_path).resolve()
        self.env_name = dataset_name
        self.verbose = verbose
        self.scene_dt = scene_dt

        # Cosmos-Drive-Dreams 风格的 HDMap 配置（可被外部覆盖）
        self.hdmap_color_config: Dict[str, List[int]] = (
            hdmap_color_config or self.DEFAULT_COSMOS_HD_MAP_COLOR_CONFIG.copy()
        )
        self.hdmap_type_config: Dict[str, int] = (
            hdmap_type_config or self.DEFAULT_COSMOS_HD_MAP_TYPE_CONFIG.copy()
        )

        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"InterHub cache not found at: {self.cache_path}\n"
                f"Please ensure 0_data_unify.py has generated data/1_unified_cache"
            )

        if self.verbose:
            print(f"[Bridge] Loading InterHub dataset from: {self.cache_path}")
            print(f"[Bridge] Environment name: {self.env_name}")

        try:
            self.env_cache = EnvCache(self.cache_path)
            self.scenes_list = self.env_cache.load_env_scenes_list(self.env_name)
            self.num_scenes = len(self.scenes_list)

            if self.verbose:
                print(
                    f"✓ Successfully loaded {self.num_scenes} scenes "
                    f"from environment '{self.env_name}'"
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load scenes from EnvCache: {e}\n"
                f"Please check path {self.cache_path}/{self.env_name}/scenes_list.dill exists"
            )

    def generate_rds_hq_from_scene(
        self,
        scene_idx: int,
        time_start: float,
        time_end: float,
        output_dir: Union[str, Path],
        ego_agent_id: Optional[str] = None,
        streetview_retrieval: bool = False,
        agent_clip_distance: float = 80.0,
        map_clip_distance: float = 100.0,
        camera_setting: str = "default",
    ) -> Path:
        """
        Generate RDS-HQ from an InterHub scene.

        Args:
            scene_idx: Scene index (0 to num_scenes-1)
            time_start: Start time in seconds
            time_end: End time in seconds
            output_dir: Output directory
            ego_agent_id: Optional ego vehicle agent ID (auto-selected if None)
            streetview_retrieval: Enable street view retrieval
            agent_clip_distance: Include agents within this radius of ego (meters)
            map_clip_distance: Map clipping distance for RDS-HQ (meters)
            camera_setting: Camera configuration preset ("default" or "waymo")

        Returns:
            Path to generated RDS-HQ directory
        """
        if scene_idx < 0 or scene_idx >= self.num_scenes:
            raise ValueError(
                f"Scene index {scene_idx} out of bounds. "
                f"Environment has {self.num_scenes} scenes (0 to {self.num_scenes-1})"
            )

        if time_end <= time_start:
            raise ValueError(
                f"Invalid time window: end time ({time_end}) "
                f"must be greater than start time ({time_start})"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        fcd_dir = output_path / "fcd"
        map_dir = output_path / "maps"
        rds_base_dir = output_path / "rds_hq"

        for d in (fcd_dir, map_dir, rds_base_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ===== STEP 1: Load Scene from InterHub Cache =====
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 1/4: Extracting Scene Data from InterHub")
            print("=" * 70)

        scene_tag = self.scenes_list[scene_idx]
        scene_name = getattr(scene_tag, "name", f"scene_{scene_idx}")

        scene: Scene = self.env_cache.load_scene(
            self.env_name, scene_name, scene_dt=self.scene_dt
        )

        if self.verbose:
            print(f"[Bridge] Scene index: {scene_idx}")
            print(f"[Bridge] Scene name:  {scene.name}")
            print(f"[Bridge] Scene dt:    {scene.dt}")
            scene_duration = scene.length_timesteps * scene.dt
            print(
                f"[Bridge] Scene duration: "
                f"{scene_duration:.2f} s "
                f"({scene.length_timesteps} timesteps)"
            )
            print(
                f"[Bridge] Requested time window: "
                f"[{time_start:.2f}, {time_end:.2f}] s"
            )

        # Clip time window to scene duration
        scene_duration = scene.length_timesteps * scene.dt
        orig_start, orig_end = time_start, time_end
        time_start = max(0.0, time_start)
        time_end = min(time_end, scene_duration)

        if time_end <= time_start:
            if self.verbose:
                print(
                    f"[Bridge] WARNING: requested window [{orig_start:.2f}, {orig_end:.2f}] "
                    f"outside scene duration [0.00, {scene_duration:.2f}]. "
                    f"Using full scene instead."
                )
            time_start = 0.0
            time_end = scene_duration

        if self.verbose:
            print(
                f"[Bridge] Effective time window: "
                f"[{time_start:.2f}, {time_end:.2f}] s"
            )

        # ===== STEP 2: Generate SUMO FCD with complete vehicle data =====
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 2/4: Generating Complete SUMO FCD File")
            print("=" * 70)

        fcd_path, ego_used, trajectory_bounds = self._generate_sumo_fcd(
            scene=scene,
            time_start=time_start,
            time_end=time_end,
            output_dir=fcd_dir,
            ego_agent_id=ego_agent_id,
            agent_clip_distance=agent_clip_distance,
        )

        if self.verbose:
            print(f"✓ FCD file created: {fcd_path}")
            print(f"  File size: {fcd_path.stat().st_size / 1024:.2f} KB")
            print(f"[Bridge] Ego agent: {ego_used}")
            print(f"[Bridge] Trajectory bounds: {trajectory_bounds}")

        # ===== STEP 3: Generate SUMO map enriched by InterHub trajectories =====
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 3/4: Creating SUMO Map from Trajectory Data")
            print("=" * 70)

        map_path = self._create_map_from_trajectories(
            scene=scene,
            trajectory_bounds=trajectory_bounds,
            output_dir=map_dir,
            fcd_path=fcd_path,
        )

        if self.verbose:
            print(f"✓ Map file created: {map_path}")

        # ===== STEP 4: Run TeraSim RDS-HQ Converter =====
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 4/4: Running TeraSim RDS-HQ Converter")
            print("=" * 70)

        # 推导与 Cosmos-Drive-Dreams 对齐的视频配置（30 FPS, 704×1280）
        duration_seconds = float(time_end - time_start)
        target_fps = 30.0
        target_num_frames = max(1, int(round(duration_seconds * target_fps)))

        cosmos_video_config = {
            "target_fps": target_fps,
            "num_frames": target_num_frames,
            "resolution_hw": [704, 1280],
        }

        cosmos_config = {
            "path_to_output": str(rds_base_dir),
            "path_to_fcd": str(fcd_path),
            "path_to_map": str(map_path),
            "camera_setting_name": camera_setting,
            "vehicle_id": ego_used,
            "time_start": float(time_start),
            "time_end": float(time_end),
            "agent_clip_distance": float(agent_clip_distance),
            "map_clip_distance": float(map_clip_distance),
            "streetview_retrieval": bool(streetview_retrieval),
            # Cosmos-Drive-Dreams 相关配置（由 TeraSim 侧解释使用）
            "hdmap_color_config": self.hdmap_color_config,
            "hdmap_type_config": self.hdmap_type_config,
            "cosmos_input_standard": "cosmos-drive-dreams-v1",
            "cosmos_video_config": cosmos_video_config,
            "interhub_metadata": {
                "env_name": self.env_name,
                "scene_name": scene.name,
                "scene_idx": int(scene_idx),
            },
        }

        cosmos_converter = TeraSimToCosmosConverter(config_dict=cosmos_config)
        cosmos_converter.convert()

        final_rds_dir = cosmos_converter.path_to_output

        if self.verbose:
            print("\n✓ RDS-HQ generation complete!")
            print(f"Output directory: {final_rds_dir}")

        return final_rds_dir

    def _generate_sumo_fcd(
        self,
        scene: Scene,
        time_start: float,
        time_end: float,
        output_dir: Path,
        ego_agent_id: Optional[str],
        agent_clip_distance: float,
    ) -> Tuple[Path, str, Dict]:
        """
        Generate SUMO FCD XML with complete vehicle attributes.

        Key improvements:
        - Adds speed attribute (calculated from velocity)
        - Adds z-coordinate (height)
        - Adds vehicle dimensions (length, width, height)
        - Tracks trajectory bounds for map generation

        Returns:
            (fcd_path, ego_agent_id, trajectory_bounds)
        """
        dt = scene.dt
        start_idx = max(0, int(time_start / dt))
        end_idx = min(scene.length_timesteps, int(time_end / dt))

        if end_idx <= start_idx:
            raise ValueError(
                f"Time window [{time_start}, {time_end}] is empty after "
                f"discretization with dt={dt}"
            )

        scene_cache = DataFrameCache(
            cache_path=self.cache_path,
            scene=scene,
        )

        # Get column indices
        col = scene_cache.column_dict
        x_idx = col["x"]
        y_idx = col["y"]

        # Find heading column (could be heading, yaw, theta, or phi)
        heading_idx = col.get("heading", None)
        if heading_idx is None:
            for key in ["yaw", "theta", "phi"]:
                if key in col:
                    heading_idx = col[key]
                    break
        if heading_idx is None:
            heading_idx = 0  # Default to 0 if no heading found

        # Find velocity columns
        vx_idx = col.get("vx", None)
        vy_idx = col.get("vy", None)

        # Find extent columns (vehicle dimensions)
        length_idx = col.get("length", None)
        width_idx = col.get("width", None)
        height_idx = col.get("height", None)

        if ego_agent_id is None:
            ego_agent_id = self._select_ego_agent(scene, start_idx, end_idx)

        if self.verbose:
            print(f"[Bridge] Ego agent: {ego_agent_id}")
            print(f"[Bridge] Agent clip distance: {agent_clip_distance} m")

        # Track trajectory bounds for map generation
        all_x: List[float] = []
        all_y: List[float] = []

        fcd_root = ET.Element("fcd-export")
        last_ego_pos = None

        for t_idx in tqdm(
            range(start_idx, end_idx),
            disable=not self.verbose,
            desc="Generating FCD",
        ):
            timestamp = t_idx * dt
            timestep_elem = ET.SubElement(
                fcd_root,
                "timestep",
                time=f"{timestamp:.2f}",
            )

            agents_present = scene.agent_presence[t_idx]
            names_present = [a.name for a in agents_present]

            # Get ego position
            ego_pos = None
            if ego_agent_id in names_present:
                try:
                    ego_state = scene_cache.get_raw_state(
                        agent_id=ego_agent_id, scene_ts=t_idx
                    )
                    ego_pos = np.array(
                        [ego_state[x_idx], ego_state[y_idx]], dtype=float
                    )
                    last_ego_pos = ego_pos
                except Exception:
                    ego_pos = last_ego_pos
            else:
                ego_pos = last_ego_pos

            if ego_pos is None:
                continue

            # Process all agents in this timestep
            for agent_meta in agents_present:
                agent_name = agent_meta.name

                # Get agent type from scene.agents dictionary
                agent_type = AgentType.VEHICLE  # Default to vehicle
                if agent_name in scene.agents:
                    agent_info = scene.agents[agent_name]
                    if hasattr(agent_info, "agent_type"):
                        agent_type = agent_info.agent_type

                try:
                    raw_state = scene_cache.get_raw_state(
                        agent_id=agent_name, scene_ts=t_idx
                    )
                except Exception:
                    continue

                # Extract position
                pos = np.array(
                    [raw_state[x_idx], raw_state[y_idx]], dtype=float
                )

                # Track bounds
                all_x.append(pos[0])
                all_y.append(pos[1])

                # Check distance from ego
                rel_dist = np.linalg.norm(pos - ego_pos)
                if rel_dist > agent_clip_distance:
                    continue

                # Calculate speed and heading (prefer velocity-based to match TeraSim/SUMO)
                spspeed = 0.0
                angle_deg = 0.0

                if vx_idx is not None and vy_idx is not None:
                    vx = float(raw_state[vx_idx])
                    vy = float(raw_state[vy_idx])
                    speed = float(math.hypot(vx, vy))

                    if speed > 1e-3:
                        # 数学坐标下的角度：0 在 +X(东)，逆时针增大；atan2(y, x)
                        theta_math_deg = math.degrees(math.atan2(vy, vx))
                        # 转成 SUMO 约定：0 在北(+Y)，顺时针增大
                        angle_deg = (90.0 - theta_math_deg) % 360.0

                # 如果速度太小或没有 vx/vy，就退回用 heading（同样做坐标系转换）
                if speed <= 1e-3 and heading_idx is not None:
                    heading_val = float(raw_state[heading_idx])
                    # 这里假定 heading 也是数学坐标系：0 在 +X，逆时针
                    heading_math_deg = math.degrees(heading_val)
                    angle_deg = (90.0 - heading_math_deg) % 360.0


                # Get vehicle dimensions
                dims = self.VEHICLE_DIMENSIONS.get(
                    agent_type, self.VEHICLE_DIMENSIONS[AgentType.VEHICLE]
                )

                if length_idx is not None:
                    length = float(raw_state[length_idx])
                else:
                    length = dims["length"]

                if width_idx is not None:
                    width = float(raw_state[width_idx])
                else:
                    width = dims["width"]

                if height_idx is not None:
                    height = float(raw_state[height_idx])
                else:
                    height = dims["height"]

                # Map agent type to SUMO type string
                if agent_type == AgentType.PEDESTRIAN:
                    # Use person element for pedestrians
                    ET.SubElement(
                        timestep_elem,
                        "person",
                        id=str(agent_name),
                        x=f"{pos[0]:.2f}",
                        y=f"{pos[1]:.2f}",
                        z=f"{height/2:.2f}",  # z at center height
                        angle=f"{angle_deg:.2f}",
                        speed=f"{speed:.2f}",
                        type="pedestrian",
                    )
                else:
                    # Use vehicle element for all other types
                    sumo_type = "passenger"
                    if agent_type == AgentType.BICYCLE:
                        sumo_type = "bicycle"
                    elif agent_type == AgentType.MOTORCYCLE:
                        sumo_type = "motorcycle"

                    ET.SubElement(
                        timestep_elem,
                        "vehicle",
                        id=str(agent_name),
                        x=f"{pos[0]:.2f}",
                        y=f"{pos[1]:.2f}",
                        z=f"{height/2:.2f}",  # z at center height
                        angle=f"{angle_deg:.2f}",
                        speed=f"{speed:.2f}",
                        type=sumo_type,
                        length=f"{length:.2f}",
                        width=f"{width:.2f}",
                        height=f"{height:.2f}",
                    )

        # Calculate trajectory bounds
        trajectory_bounds = {
            "min_x": float(np.min(all_x)),
            "max_x": float(np.max(all_x)),
            "min_y": float(np.min(all_y)),
            "max_y": float(np.max(all_y)),
        }

        # Write FCD file
        fcd_path = output_dir / f"{scene.name}_fcd.xml"
        rough_string = ET.tostring(fcd_root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(fcd_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        return fcd_path, ego_agent_id, trajectory_bounds

    def _create_map_from_trajectories(
        self,
        scene: Scene,
        trajectory_bounds: Dict,
        output_dir: Path,
        fcd_path: Optional[Path] = None,
    ) -> Path:
        """
        Create a SUMO network map that covers the trajectory bounds.

        基础部分：
            生成一个规则的栅格路网，保证无论怎样都有可用的地图。
        增强部分（如果提供 fcd_path）：
            解析 FCD 中的真实车辆轨迹，为位移足够大的车辆生成额外 edge/lane，
            让道路结构更贴近 InterHub 轨迹分布。

        Args:
            scene: Scene object
            trajectory_bounds: Dict with min_x, max_x, min_y, max_y
            output_dir: Output directory
            fcd_path: optional path to FCD file for trajectory-based enrichment

        Returns:
            Path to created map file
        """
        map_path = output_dir / f"{scene.name}_map.net.xml"

        # Add padding to bounds
        padding = 50.0  # meters
        min_x = trajectory_bounds["min_x"] - padding
        max_x = trajectory_bounds["max_x"] + padding
        min_y = trajectory_bounds["min_y"] - padding
        max_y = trajectory_bounds["max_y"] + padding

        # Calculate center and dimensions
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y

        if self.verbose:
            print(
                f"[Bridge] Map bounds: X=[{min_x:.1f}, {max_x:.1f}], "
                f"Y=[{min_y:.1f}, {max_y:.1f}]"
            )
            print(f"[Bridge] Map center: ({center_x:.1f}, {center_y:.1f})")
            print(f"[Bridge] Map size: {width:.1f}m x {height:.1f}m")

        # Create SUMO network with proper version
        net_root = ET.Element("net", {"version": "1.16"})

        # Location element with actual bounds
        ET.SubElement(
            net_root,
            "location",
            {
                "netOffset": f"{center_x:.2f},{center_y:.2f}",
                "convBoundary": f"{min_x:.2f},{min_y:.2f},{max_x:.2f},{max_y:.2f}",
                "origBoundary": f"{min_x:.2f},{min_y:.2f},{max_x:.2f},{max_y:.2f}",
                "projParameter": "!",
            },
        )

        # ------------------------------------------------------------------
        # 1) 规则栅格路网（保证稳定性 & 全覆盖）
        # ------------------------------------------------------------------
        num_horizontal = 5
        num_vertical = 5

        # Create horizontal edges (west-east)
        for i in range(num_horizontal):
            y_pos = min_y + (i / (num_horizontal - 1)) * height

            from_node = f"n_h{i}_0"
            to_node = f"n_h{i}_1"
            edge_id_str = f"edge_h{i}"

            # Create edge
            edge = ET.SubElement(
                net_root,
                "edge",
                {
                    "id": edge_id_str,
                    "from": from_node,
                    "to": to_node,
                },
            )

            # Add lane
            ET.SubElement(
                edge,
                "lane",
                {
                    "id": f"{edge_id_str}_0",
                    "index": "0",
                    "speed": "13.89",
                    "length": f"{width:.2f}",
                    "shape": f"{min_x:.2f},{y_pos:.2f} {max_x:.2f},{y_pos:.2f}",
                },
            )

        # Create vertical edges (south-north)
        for j in range(num_vertical):
            x_pos = min_x + (j / (num_vertical - 1)) * width

            from_node = f"n_v{j}_0"
            to_node = f"n_v{j}_1"
            edge_id_str = f"edge_v{j}"

            # Create edge
            edge = ET.SubElement(
                net_root,
                "edge",
                {
                    "id": edge_id_str,
                    "from": from_node,
                    "to": to_node,
                },
            )

            # Add lane
            ET.SubElement(
                edge,
                "lane",
                {
                    "id": f"{edge_id_str}_0",
                    "index": "0",
                    "speed": "13.89",
                    "length": f"{height:.2f}",
                    "shape": f"{x_pos:.2f},{min_y:.2f} {x_pos:.2f},{max_y:.2f}",
                },
            )

        # Create junctions at intersections
        # Horizontal edge junctions
        for i in range(num_horizontal):
            y_pos = min_y + (i / (num_horizontal - 1)) * height

            # West junction
            ET.SubElement(
                net_root,
                "junction",
                {
                    "id": f"n_h{i}_0",
                    "type": "priority",
                    "x": f"{min_x:.2f}",
                    "y": f"{y_pos:.2f}",
                    "incLanes": "",
                    "intLanes": "",
                    "shape": f"{min_x:.2f},{y_pos:.2f}",
                },
            )

            # East junction
            ET.SubElement(
                net_root,
                "junction",
                {
                    "id": f"n_h{i}_1",
                    "type": "priority",
                    "x": f"{max_x:.2f}",
                    "y": f"{y_pos:.2f}",
                    "incLanes": f"edge_h{i}_0",
                    "intLanes": "",
                    "shape": f"{max_x:.2f},{y_pos:.2f}",
                },
            )

        # Vertical edge junctions
        for j in range(num_vertical):
            x_pos = min_x + (j / (num_vertical - 1)) * width

            # South junction
            ET.SubElement(
                net_root,
                "junction",
                {
                    "id": f"n_v{j}_0",
                    "type": "priority",
                    "x": f"{x_pos:.2f}",
                    "y": f"{min_y:.2f}",
                    "incLanes": "",
                    "intLanes": "",
                    "shape": f"{x_pos:.2f},{min_y:.2f}",
                },
            )

            # North junction
            ET.SubElement(
                net_root,
                "junction",
                {
                    "id": f"n_v{j}_1",
                    "type": "priority",
                    "x": f"{x_pos:.2f}",
                    "y": f"{max_y:.2f}",
                    "incLanes": f"edge_v{j}_0",
                    "intLanes": "",
                    "shape": f"{x_pos:.2f},{max_y:.2f}",
                },
            )

        # ------------------------------------------------------------------
        # 2) 使用 FCD 轨迹进一步“加密”路网（如果可用）
        # ------------------------------------------------------------------
        if fcd_path is not None and fcd_path.exists():
            if self.verbose:
                print(f"[Bridge] Enriching SUMO map using trajectories from FCD: {fcd_path}")
            try:
                self._add_trajectory_edges_to_net(net_root=net_root, fcd_path=fcd_path)
            except Exception as e:
                if self.verbose:
                    print(f"[Bridge] WARNING: failed to enrich map from FCD: {e}")

        # Write map file
        rough_string = ET.tostring(net_root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(map_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        return map_path

    def _add_trajectory_edges_to_net(
        self,
        net_root: ET.Element,
        fcd_path: Path,
        min_displacement: float = 10.0,
        max_agents: int = 64,
        max_points_per_traj: int = 32,
    ) -> None:
        """
        从 FCD 文件中解析各车辆轨迹，并将“位移足够大”的轨迹转为 SUMO edge/lane。

        这样生成的 HDMap 不再是完全规则的栅格，而是叠加了真实车流走向，
        更符合 Cosmos-Drive-Dreams 对精确布局控制的需求。

        Args:
            net_root: SUMO net 根节点
            fcd_path: 已生成的 FCD XML 路径
            min_displacement: 仅为总位移大于该阈值的车辆生成 edge
            max_agents: 最多转换多少辆车的轨迹
            max_points_per_traj: 每条轨迹最多采样多少个点用于 lane.shape
        """
        tree = ET.parse(str(fcd_path))
        root = tree.getroot()

        # 收集每台车的 (time, x, y) 序列
        vehicle_trajs: Dict[str, List[Tuple[float, float, float]]] = {}

        for timestep in root.findall("timestep"):
            t = float(timestep.get("time", "0.0"))
            for veh in timestep.findall("vehicle"):
                vid = veh.get("id")
                if vid is None:
                    continue
                x = float(veh.get("x", "0.0"))
                y = float(veh.get("y", "0.0"))

                vehicle_trajs.setdefault(vid, []).append((t, x, y))

        if not vehicle_trajs:
            if self.verbose:
                print("[Bridge] No vehicle trajectories found in FCD; skip enrichment.")
            return

        # 计算每辆车的总位移
        def compute_displacement(traj: List[Tuple[float, float, float]]) -> float:
            if len(traj) < 2:
                return 0.0
            disp = 0.0
            for i in range(1, len(traj)):
                _, x0, y0 = traj[i - 1]
                _, x1, y1 = traj[i]
                dx = x1 - x0
                dy = y1 - y0
                disp += float(np.sqrt(dx * dx + dy * dy))
            return disp

        agents_stats: List[Tuple[str, float]] = []
        for vid, traj in vehicle_trajs.items():
            # 按时间排序
            traj_sorted = sorted(traj, key=lambda p: p[0])
            disp = compute_displacement(traj_sorted)
            if disp >= min_displacement:
                agents_stats.append((vid, disp))

        if not agents_stats:
            if self.verbose:
                print(
                    f"[Bridge] No trajectories exceed min_displacement={min_displacement:.1f}m; skip enrichment."
                )
            return

        # 选取位移最大的若干辆车
        agents_stats.sort(key=lambda x: x[1], reverse=True)
        selected_agents = agents_stats[:max_agents]

        if self.verbose:
            print(
                f"[Bridge] Using {len(selected_agents)} trajectories "
                f"to enrich SUMO net (min_displacement={min_displacement:.1f}m)."
            )

        # 收集已有 junction id，避免重复创建
        existing_junction_ids = {
            junc.get("id") for junc in net_root.findall("junction")
        }

        for vid, disp in selected_agents:
            traj = sorted(vehicle_trajs[vid], key=lambda p: p[0])
            if len(traj) < 2:
                continue

            # 轨迹总长度
            length = 0.0
            for i in range(1, len(traj)):
                _, x0, y0 = traj[i - 1]
                _, x1, y1 = traj[i]
                dx = x1 - x0
                dy = y1 - y0
                length += float(np.sqrt(dx * dx + dy * dy))
            if length < 1.0:
                continue

            # 下采样轨迹点用于 lane.shape
            step = max(1, len(traj) // max_points_per_traj)
            sampled = traj[::step]
            if sampled[-1] != traj[-1]:
                sampled.append(traj[-1])

            shape_str = " ".join(f"{x:.2f},{y:.2f}" for _, x, y in sampled)

            # 节点位置：用起点和终点
            _, sx, sy = sampled[0]
            _, ex, ey = sampled[-1]

            # 清洗车辆 id 以构造合法的 edge/junction id
            safe_vid = re.sub(r"[^a-zA-Z0-9_]", "_", str(vid))
            start_node_id = f"traj_{safe_vid}_s"
            end_node_id = f"traj_{safe_vid}_e"
            edge_id = f"traj_{safe_vid}"

            # 创建（或复用）起点/终点 junction
            if start_node_id not in existing_junction_ids:
                ET.SubElement(
                    net_root,
                    "junction",
                    {
                        "id": start_node_id,
                        "type": "priority",
                        "x": f"{sx:.2f}",
                        "y": f"{sy:.2f}",
                        "incLanes": "",
                        "intLanes": "",
                        "shape": f"{sx:.2f},{sy:.2f}",
                    },
                )
                existing_junction_ids.add(start_node_id)

            if end_node_id not in existing_junction_ids:
                ET.SubElement(
                    net_root,
                    "junction",
                    {
                        "id": end_node_id,
                        "type": "priority",
                        "x": f"{ex:.2f}",
                        "y": f"{ey:.2f}",
                        "incLanes": "",
                        "intLanes": "",
                        "shape": f"{ex:.2f},{ey:.2f}",
                    },
                )
                existing_junction_ids.add(end_node_id)

            # 创建基于真实轨迹的 edge + lane
            edge_elem = ET.SubElement(
                net_root,
                "edge",
                {
                    "id": edge_id,
                    "from": start_node_id,
                    "to": end_node_id,
                },
            )

            # 速度这里用一个较为保守的默认值（50 km/h）
            ET.SubElement(
                edge_elem,
                "lane",
                {
                    "id": f"{edge_id}_0",
                    "index": "0",
                    "speed": "13.89",
                    "length": f"{length:.2f}",
                    "shape": shape_str,
                },
            )

    def _select_ego_agent(
        self,
        scene: Scene,
        start_idx: int,
        end_idx: int,
    ) -> str:
        """
        Select an ego vehicle from the scene.

        Prioritizes:
        1. Vehicles present throughout the entire time window
        2. VEHICLE type agents over others
        3. First available agent as fallback
        """
        # Find agents present in all timesteps
        agents_all_present = None

        for t_idx in range(start_idx, end_idx):
            agents_at_t = set([a.name for a in scene.agent_presence[t_idx]])
            if agents_all_present is None:
                agents_all_present = agents_at_t
            else:
                agents_all_present = agents_all_present.intersection(agents_at_t)

        if not agents_all_present:
            # Fallback: use most frequently present agent
            agent_counts: Dict[str, int] = {}
            for t_idx in range(start_idx, end_idx):
                for agent_meta in scene.agent_presence[t_idx]:
                    agent_counts[agent_meta.name] = (
                        agent_counts.get(agent_meta.name, 0) + 1
                    )

            if agent_counts:
                ego_id = max(agent_counts, key=agent_counts.get)
                if self.verbose:
                    print(
                        "[Bridge] WARNING: No agent present in all timesteps. "
                        f"Using most frequent: {ego_id}"
                    )
                return ego_id
            else:
                raise ValueError("No agents found in specified time window")

        # Try to find a vehicle type by checking scene.agents dictionary
        for agent_name in agents_all_present:
            # Access agent info from scene.agents dictionary
            if agent_name in scene.agents:
                agent_info = scene.agents[agent_name]
                # Check if this agent is a vehicle
                if (
                    hasattr(agent_info, "agent_type")
                    and agent_info.agent_type == AgentType.VEHICLE
                ):
                    return agent_name

        # Return first agent if no vehicle found
        return list(agents_all_present)[0]
    def _render_scene_preview_image(
        self,
        scene: Scene,
        bounds: Dict[str, float],
        vec_map=None,
        snapshot_time: Optional[float] = None,
    ) -> np.ndarray:
        """
        渲染一张静态场景缩略图：
        - 背景：路网轮廓
        - 前景：矩形车辆（在 snapshot_time 时刻）

        返回值：
            image: (H, W, 3) 的 uint8 RGB 图像数组
        """
        import matplotlib.pyplot as plt

        dt = scene.dt
        scene_duration = scene.length_timesteps * dt

        if snapshot_time is None:
            snapshot_time = scene_duration * 0.5

        snapshot_time = max(0.0, min(snapshot_time, max(0.0, scene_duration - dt)))
        t_idx = int(snapshot_time / dt)

        scene_cache = DataFrameCache(
            cache_path=self.cache_path,
            scene=scene,
        )
        col = scene_cache.column_dict
        x_idx = col["x"]
        y_idx = col["y"]
        heading_idx = col.get("heading")
        if heading_idx is None:
            for key in ["yaw", "theta", "phi"]:
                if key in col:
                    heading_idx = col[key]
                    break
        length_idx = col.get("length")
        width_idx = col.get("width")

        # margin 和 BEV 一致
        margin_x = max(5.0, 0.1 * max(1e-3, bounds["max_x"] - bounds["min_x"]))
        margin_y = max(5.0, 0.1 * max(1e-3, bounds["max_y"] - bounds["min_y"]))

        if vec_map is None:
            vec_map = self._get_vec_map_for_scene(scene)

        agents_present = scene.agent_presence[t_idx]

        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)

        # 背景路网
        self._draw_map_background(ax, vec_map, bounds)

        # 矩形车辆
        for agent_meta in agents_present:
            name = agent_meta.name
            try:
                state = scene_cache.get_raw_state(agent_id=name, scene_ts=t_idx)
            except Exception:
                continue

            x = float(state[x_idx])
            y = float(state[y_idx])

            if heading_idx is not None:
                heading = float(state[heading_idx])
            else:
                heading = 0.0

            # 车辆尺寸：优先用状态里的 length/width，其次默认
            if length_idx is not None:
                length = float(state[length_idx])
            else:
                agent_type = AgentType.VEHICLE
                if name in scene.agents and hasattr(scene.agents[name], "agent_type"):
                    agent_type = scene.agents[name].agent_type
                length = self.VEHICLE_DIMENSIONS.get(
                    agent_type, self.VEHICLE_DIMENSIONS[AgentType.VEHICLE]
                )["length"]

            if width_idx is not None:
                width = float(state[width_idx])
            else:
                agent_type = AgentType.VEHICLE
                if name in scene.agents and hasattr(scene.agents[name], "agent_type"):
                    agent_type = scene.agents[name].agent_type
                width = self.VEHICLE_DIMENSIONS.get(
                    agent_type, self.VEHICLE_DIMENSIONS[AgentType.VEHICLE]
                )["width"]

            poly = self._vehicle_polygon(x, y, heading, length=length, width=width)
            rect = patches.Polygon(
                poly,
                closed=True,
                zorder=10,
                facecolor="#3868A6",
                edgecolor="none",
                alpha=0.9,
            )
            ax.add_patch(rect)

        ax.set_aspect("equal", "box")
        ax.set_xlim(bounds["min_x"] - margin_x, bounds["max_x"] + margin_x)
        ax.set_ylim(bounds["min_y"] - margin_y, bounds["max_y"] + margin_y)
        ax.axis("off")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)
        image = buf[..., :3].copy()
        plt.close(fig)

        return image

    def _get_vec_map_for_scene(self, scene: Scene):
        """
        使用 InterHub 同款方式，从 trajdata MapAPI 里拿当前场景的矢量路网。
        """
        try:
            map_api = MapAPI(self.cache_path)
            env_name = getattr(scene, "env_name", self.env_name)
            location = getattr(scene, "location", None)
            if location is None:
                if self.verbose:
                    print(f"[Bridge] WARNING: scene '{scene.name}' has no location; skip map.")
                return None
            vec_map = map_api.get_map(f"{env_name}:{location}")
            return vec_map
        except Exception as e:
            if self.verbose:
                print(f"[Bridge] WARNING: failed to load vec_map for scene '{scene.name}': {e}")
            return None

    def _is_line_in_view(self, line_points, view_rect):
        """
        简化版 is_line_in_view：用 shapely 判断 lane polyline 是否和视野矩形相交。
        """
        try:
            if line_points is None or len(line_points) == 0:
                return False
            line = LineString(line_points)
            return line.intersects(view_rect)
        except Exception:
            return False

    def _draw_map_background(self, ax, vec_map, bounds: Dict[str, float]):
        """
        在给定坐标范围内绘制路网轮廓（左右车道边缘），类似 InterHub 的 plot_lanes。
        """
        if vec_map is None:
            return

        view_rect = shapely_box(
            bounds["min_x"],
            bounds["min_y"],
            bounds["max_x"],
            bounds["max_y"],
        )

        for lane in vec_map.lanes:
            left_edge = lane.left_edge
            right_edge = lane.right_edge

            if left_edge is not None and self._is_line_in_view(left_edge.points, view_rect):
                ax.plot(
                    left_edge.points[:, 0],
                    left_edge.points[:, 1],
                    "-",
                    linewidth=0.4,
                    color="#969696",
                    zorder=1,
                )

            if right_edge is not None and self._is_line_in_view(right_edge.points, view_rect):
                ax.plot(
                    right_edge.points[:, 0],
                    right_edge.points[:, 1],
                    "-",
                    linewidth=0.4,
                    color="#969696",
                    zorder=1,
                )

    def _rotate_around_center(self, pts: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
        """
        和 InterHub 中 rotate_around_center 一样的旋转逻辑。
        """
        shifted = pts - center
        R = np.array(
            [
                [math.cos(yaw), math.sin(yaw)],
                [-math.sin(yaw), math.cos(yaw)],
            ]
        )
        return shifted @ R + center

    def _vehicle_polygon(
        self,
        x: float,
        y: float,
        psi_rad: float,
        length: float,
        width: float,
    ) -> np.ndarray:
        """
        参考 InterHub 的 polygon_xy_from_motionstate：
        根据车辆中心、航向角、长宽生成矩形四个顶点。
        """
        lowleft = (x - length / 2.0, y - width / 2.0)
        lowright = (x + length / 2.0, y - width / 2.0)
        upright = (x + length / 2.0, y + width / 2.0)
        upleft = (x - length / 2.0, y + width / 2.0)

        corners = np.array([lowleft, lowright, upright, upleft], dtype=float)

        # 和 InterHub 一样，轻微减去一个固定角度（1°），避免坐标系差异造成的视觉偏差
        yaw_for_draw = psi_rad - math.pi / 180.0
        rotated = self._rotate_around_center(corners, np.array([x, y], dtype=float), yaw_for_draw)
        return rotated

    # ------------------------------------------------------------------
    # 新增：BEV + GUI 选车 + 生成 RDS-HQ
    # ------------------------------------------------------------------
    def _generate_scene_gif(
        self,
        scene: Scene,
        time_start: float,
        time_end: float,
        bounds: Dict[str, float],
        output_dir: Union[str, Path],
        max_frames: int = 60,
        fps: int = 5,
        vec_map=None,
    ) -> Path:
        """
        使用 InterHub 场景数据生成一个简易鸟瞰 GIF：
        - 背景：路网轮廓（左右车道边缘）
        - 前景：用矩形表示场景中的车辆

        Args:
            scene: InterHub Scene
            time_start, time_end: 想要展示的时间窗口
            bounds: 用于绘图的 XY 范围（和 BEV 一致）
            output_dir: GIF 输出目录
            max_frames: GIF 最多包含的帧数
            fps: GIF 播放帧率
            vec_map: 已经加载好的 VectorMap（可选，不传则内部再加载一次）

        Returns:
            gif_path: 生成的 GIF 文件路径
        """
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dt = scene.dt
        scene_duration = scene.length_timesteps * dt

        # 把时间窗口 clamp 到场景范围内
        time_start = max(0.0, min(time_start, scene_duration))
        time_end = max(0.0, min(time_end, scene_duration))
        if time_end <= time_start:
            time_start = 0.0
            time_end = scene_duration

        start_idx = max(0, int(time_start / dt))
        end_idx = min(scene.length_timesteps, int(time_end / dt))

        if end_idx <= start_idx:
            raise ValueError(
                f"Invalid time window [{time_start}, {time_end}] for GIF generation."
            )

        scene_cache = DataFrameCache(
            cache_path=self.cache_path,
            scene=scene,
        )
        col = scene_cache.column_dict
        if "x" not in col or "y" not in col:
            raise KeyError("DataFrameCache.column_dict must contain 'x' and 'y' columns")

        x_idx = col["x"]
        y_idx = col["y"]
        heading_idx = col.get("heading")
        if heading_idx is None:
            for key in ["yaw", "theta", "phi"]:
                if key in col:
                    heading_idx = col[key]
                    break
        length_idx = col.get("length")
        width_idx = col.get("width")

        # 选一些时间步来画（最多 max_frames 帧）
        indices = list(range(start_idx, end_idx))
        if len(indices) > max_frames:
            step = math.ceil(len(indices) / max_frames)
            indices = indices[::step]

        frames: List[np.ndarray] = []

        # margin 保持和 BEV 一致
        margin_x = max(5.0, 0.1 * max(1e-3, bounds["max_x"] - bounds["min_x"]))
        margin_y = max(5.0, 0.1 * max(1e-3, bounds["max_y"] - bounds["min_y"]))

        # 没传 vec_map 就自己加载一下
        if vec_map is None:
            vec_map = self._get_vec_map_for_scene(scene)

        for t_idx in indices:
            agents_present = scene.agent_presence[t_idx]

            fig, ax = plt.subplots(figsize=(4, 4))

            # 背景路网
            self._draw_map_background(ax, vec_map, bounds)

            # 绘制每辆车的矩形
            for agent_meta in agents_present:
                name = agent_meta.name
                try:
                    state = scene_cache.get_raw_state(agent_id=name, scene_ts=t_idx)
                except Exception:
                    continue

                x = float(state[x_idx])
                y = float(state[y_idx])

                if heading_idx is not None:
                    heading = float(state[heading_idx])
                else:
                    heading = 0.0

                # 车辆尺寸：优先用状态里的 length/width，没有就走默认
                if length_idx is not None:
                    length = float(state[length_idx])
                else:
                    agent_type = AgentType.VEHICLE
                    if name in scene.agents and hasattr(scene.agents[name], "agent_type"):
                        agent_type = scene.agents[name].agent_type
                    length = self.VEHICLE_DIMENSIONS.get(
                        agent_type, self.VEHICLE_DIMENSIONS[AgentType.VEHICLE]
                    )["length"]

                if width_idx is not None:
                    width = float(state[width_idx])
                else:
                    agent_type = AgentType.VEHICLE
                    if name in scene.agents and hasattr(scene.agents[name], "agent_type"):
                        agent_type = scene.agents[name].agent_type
                    width = self.VEHICLE_DIMENSIONS.get(
                        agent_type, self.VEHICLE_DIMENSIONS[AgentType.VEHICLE]
                    )["width"]

                poly = self._vehicle_polygon(x, y, heading, length=length, width=width)
                rect = patches.Polygon(
                    poly,
                    closed=True,
                    zorder=10,
                    facecolor="#3868A6",
                    edgecolor="none",
                    alpha=0.9,
                )
                ax.add_patch(rect)

            ax.set_aspect("equal", "box")
            ax.set_xlim(bounds["min_x"] - margin_x, bounds["max_x"] + margin_x)
            ax.set_ylim(bounds["min_y"] - margin_y, bounds["max_y"] + margin_y)
            ax.axis("off")
            ax.set_title(f"t = {t_idx * dt:.2f}s", fontsize=8)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(h, w, 4)     # RGBA
            image = buf[..., :3].copy()    # 转成 RGB

            frames.append(image)
            plt.close(fig)

        gif_path = output_dir / f"{scene.name}_interhub_scene.gif"
        imageio.mimsave(str(gif_path), frames, fps=fps)

        if self.verbose:
            print(f"[Bridge] InterHub scene GIF saved to: {gif_path}")

        return gif_path



    def _build_bev_for_scene(
        self,
        scene: Scene,
        snapshot_time: float,
        vehicles_only: bool = True,
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float], float]:
        """
        Build a simple bird's-eye-view (BEV) from InterHub data at a given time.

        Uses InterHub's Scene + DataFrameCache to get all agent positions
        at snapshot_time, optionally filtering to vehicles only.

        Returns:
            positions: dict agent_name -> (x, y)
            bounds: dict with min_x, max_x, min_y, max_y
            actual_snapshot_time: time (in seconds) actually used (after clamping)
        """
        dt = scene.dt
        scene_duration = scene.length_timesteps * dt

        # Clamp snapshot time into scene duration
        snapshot_time = max(0.0, min(snapshot_time, max(0.0, scene_duration - dt)))
        t_idx = int(snapshot_time / dt)

        scene_cache = DataFrameCache(
            cache_path=self.cache_path,
            scene=scene,
        )
        col = scene_cache.column_dict
        if "x" not in col or "y" not in col:
            raise KeyError("DataFrameCache.column_dict must contain 'x' and 'y' columns")

        x_idx = col["x"]
        y_idx = col["y"]

        positions: Dict[str, Tuple[float, float]] = {}
        xs: List[float] = []
        ys: List[float] = []

        agents_present = scene.agent_presence[t_idx]

        for agent_meta in agents_present:
            name = agent_meta.name
            # Optionally restrict to vehicles only
            if vehicles_only and name in scene.agents:
                info = scene.agents[name]
                if hasattr(info, "agent_type") and info.agent_type != AgentType.VEHICLE:
                    continue

            try:
                state = scene_cache.get_raw_state(agent_id=name, scene_ts=t_idx)
            except Exception:
                continue

            x = float(state[x_idx])
            y = float(state[y_idx])
            positions[name] = (x, y)
            xs.append(x)
            ys.append(y)

        if not positions:
            raise ValueError(
                "No agents found at snapshot time for BEV visualization. "
                "Try a different time window."
            )

        bounds = {
            "min_x": float(min(xs)),
            "max_x": float(max(xs)),
            "min_y": float(min(ys)),
            "max_y": float(max(ys)),
        }

        return positions, bounds, float(snapshot_time)

    def _interactive_select_vehicle_from_bev(
        self,
        positions: Dict[str, Tuple[float, float]],
        bounds: Dict[str, float],
        gif_path: Optional[Union[str, Path]] = None,
        scene: Optional[Scene] = None,
        snapshot_time: Optional[float] = None,
        vec_map=None,
        window_title: str = "InterHub Bird's-Eye View - Select Ego",
    ) -> str:
        """
        GUI 左侧：带路网轮廓 + 矩形车辆的 BEV，可点击选车（红圈高亮）。
        GUI 右侧：InterHub 场景 GIF 对照（如有）。

        返回选中的车辆 id 作为 ego。
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        # 准备 GIF 帧
        gif_frames = None
        if gif_path is not None:
            gif_path = Path(gif_path)
            if gif_path.exists():
                try:
                    gif_frames = imageio.mimread(str(gif_path))
                except Exception as e:
                    if self.verbose:
                        print(f"[Bridge] WARNING: failed to read GIF {gif_path}: {e}")
                    gif_frames = None

        # 创建窗口：有 GIF 就 1x2，没有就单图
        if gif_frames:
            fig, (ax_bev, ax_gif) = plt.subplots(1, 2, figsize=(12, 6))
        else:
            fig, ax_bev = plt.subplots(figsize=(6, 6))
            ax_gif = None

        try:
            fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass

        names: List[str] = list(positions.keys())
        xs = np.array([positions[n][0] for n in names], dtype=float)
        ys = np.array([positions[n][1] for n in names], dtype=float)

        # 先画路网轮廓
        self._draw_map_background(ax_bev, vec_map, bounds)

        # 如果有 scene + snapshot_time，用矩形画出所有车辆
        if scene is not None and snapshot_time is not None:
            dt = scene.dt
            t_idx = int(snapshot_time / dt)

            scene_cache = DataFrameCache(
                cache_path=self.cache_path,
                scene=scene,
            )
            col = scene_cache.column_dict
            x_idx = col["x"]
            y_idx = col["y"]
            heading_idx = col.get("heading")
            if heading_idx is None:
                for key in ["yaw", "theta", "phi"]:
                    if key in col:
                        heading_idx = col[key]
                        break
            length_idx = col.get("length")
            width_idx = col.get("width")

            for name in names:
                try:
                    state = scene_cache.get_raw_state(agent_id=name, scene_ts=t_idx)
                except Exception:
                    continue

                x = float(state[x_idx])
                y = float(state[y_idx])

                if heading_idx is not None:
                    heading = float(state[heading_idx])
                else:
                    heading = 0.0

                # 尺寸：优先状态里的 length/width，其次默认
                if length_idx is not None:
                    length = float(state[length_idx])
                else:
                    agent_type = AgentType.VEHICLE
                    if name in scene.agents and hasattr(scene.agents[name], "agent_type"):
                        agent_type = scene.agents[name].agent_type
                    length = self.VEHICLE_DIMENSIONS.get(
                        agent_type, self.VEHICLE_DIMENSIONS[AgentType.VEHICLE]
                    )["length"]

                if width_idx is not None:
                    width = float(state[width_idx])
                else:
                    agent_type = AgentType.VEHICLE
                    if name in scene.agents and hasattr(scene.agents[name], "agent_type"):
                        agent_type = scene.agents[name].agent_type
                    width = self.VEHICLE_DIMENSIONS.get(
                        agent_type, self.VEHICLE_DIMENSIONS[AgentType.VEHICLE]
                    )["width"]

                poly = self._vehicle_polygon(x, y, heading, length=length, width=width)
                rect = patches.Polygon(
                    poly,
                    closed=True,
                    zorder=10,
                    facecolor="#3868A6",
                    edgecolor="none",
                    alpha=0.9,
                )
                ax_bev.add_patch(rect)

        # 再画散点（方便点击命中）
        ax_bev.scatter(xs, ys, s=20, alpha=0.0, zorder=20)  # alpha=0 只用于拾取
        ax_bev.set_aspect("equal", "box")
        margin_x = max(5.0, 0.1 * max(1e-3, bounds["max_x"] - bounds["min_x"]))
        margin_y = max(5.0, 0.1 * max(1e-3, bounds["max_y"] - bounds["min_y"]))
        ax_bev.set_xlim(bounds["min_x"] - margin_x, bounds["max_x"] + margin_x)
        ax_bev.set_ylim(bounds["min_y"] - margin_y, bounds["max_y"] + margin_y)
        ax_bev.set_xlabel("X (m)")
        ax_bev.set_ylabel("Y (m)")
        ax_bev.set_title(
            "Bird's-Eye View\n"
            "Click a vehicle to select ego, then close the window."
        )

        highlight = ax_bev.scatter(
            [], [], s=120, facecolors="none", edgecolors="red", linewidths=2, zorder=30
        )
        selected_idx = {"idx": None}
        text_handle = {"obj": None}

        span = max(bounds["max_x"] - bounds["min_x"], bounds["max_y"] - bounds["min_y"])
        if span <= 0:
            span = 50.0
        select_threshold = max(2.0, 0.05 * span)

        def on_click(event):
            if event.inaxes is not ax_bev:
                return
            if event.xdata is None or event.ydata is None:
                return

            x_click = float(event.xdata)
            y_click = float(event.ydata)
            dists = np.hypot(xs - x_click, ys - y_click)
            idx = int(np.argmin(dists))
            if dists[idx] > select_threshold:
                return

            selected_idx["idx"] = idx

            highlight.set_offsets([[xs[idx], ys[idx]]])

            if text_handle["obj"] is not None:
                text_handle["obj"].remove()
            text_handle["obj"] = ax_bev.text(
                xs[idx],
                ys[idx],
                f" {names[idx]}",
                color="red",
                fontsize=9,
                fontweight="bold",
                zorder=31,
            )

            fig.canvas.draw_idle()

            if self.verbose:
                print(f"[Bridge] Selected ego candidate: {names[idx]}")

        fig.canvas.mpl_connect("button_press_event", on_click)

        # 右侧 GIF 播放
        if gif_frames and ax_gif is not None:
            ax_gif.set_title("InterHub Scene GIF")
            ax_gif.axis("off")
            im = ax_gif.imshow(gif_frames[0])

            if len(gif_frames) > 1:
                def update(frame_idx):
                    im.set_data(gif_frames[frame_idx])
                    return [im]

                ani = FuncAnimation(
                    fig,
                    update,
                    frames=len(gif_frames),
                    interval=200,
                    blit=True,
                    repeat=True,
                )
                fig._interhub_gif_anim = ani  # 防止 GC

        plt.show()  # 阻塞直到窗口关闭

        if selected_idx["idx"] is None:
            raise RuntimeError(
                "No vehicle was selected in BEV GUI. "
                "Please click near a vehicle before closing the window."
            )

        return names[selected_idx["idx"]]


    def interactive_select_scene_with_gif(
        self,
        time_start: float,
        time_end: float,
        preview_output_dir: Union[str, Path],
        max_frames: int = 60,  # 参数保留，但不再用 GIF
        fps: int = 5,          # 参数保留，但不再用 GIF
    ) -> int:
        """
        交互式选择场景（静态缩略图版本）：

        - 不再使用 GIF，只渲染一张静态预览图；
        - 所有场景以网格方式布局，类似 Windows 图标；
        - 底部有滑动条（Slider），用于翻页浏览场景；
        - 点击任意缩略图，即选中该场景并关闭窗口。

        操作方式：
          - 鼠标左键点击某个缩略图：选中该场景并关闭窗口；
          - 拖动底部滑动条：切换不同页的场景缩略图；
          - q 或 Esc：取消（抛出异常）。
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        import math

        preview_output_dir = Path(preview_output_dir)
        preview_output_dir.mkdir(parents=True, exist_ok=True)

        num_scenes = self.num_scenes
        if num_scenes == 0:
            raise ValueError("No scenes found in InterHub cache for scene selection.")

        # ------------------------------------------------------------------
        # 1) 预先为所有场景生成静态缩略图（路网 + 矩形车辆）
        # ------------------------------------------------------------------
        previews: List[Dict[str, Any]] = []

        if self.verbose:
            print(f"[Bridge] Generating static previews for {num_scenes} scenes...")

        snapshot_time = 0.5 * (time_start + time_end)

        for s_idx in range(num_scenes):
            scene_tag = self.scenes_list[s_idx]
            scene_name = getattr(scene_tag, "name", f"scene_{s_idx}")
            scene: Scene = self.env_cache.load_scene(
                self.env_name, scene_name, scene_dt=self.scene_dt
            )

            # 用 snapshot_time 生成 BEV，拿到 bounds
            positions, bounds, actual_t = self._build_bev_for_scene(
                scene=scene,
                snapshot_time=snapshot_time,
                vehicles_only=True,
            )

            vec_map = self._get_vec_map_for_scene(scene)
            img = self._render_scene_preview_image(
                scene=scene,
                bounds=bounds,
                vec_map=vec_map,
                snapshot_time=actual_t,
            )

            previews.append(
                {
                    "scene_idx": s_idx,
                    "scene_name": scene_name,
                    "image": img,
                }
            )

        if self.verbose:
            print("[Bridge] Previews generated. Launching scene selection GUI...")

        # ------------------------------------------------------------------
        # 2) 构建 GUI：网格缩略图 + 底部 Slider
        # ------------------------------------------------------------------
        # 网格布局：行列数可以按需调整
        thumbs_per_row = 4
        thumbs_per_col = 3
        thumbs_per_page = thumbs_per_row * thumbs_per_col

        num_pages = max(1, math.ceil(num_scenes / thumbs_per_page))

        fig = plt.figure(figsize=(10, 8))
        try:
            fig.canvas.manager.set_window_title("Select Scene (click thumbnail)")
        except Exception:
            pass

        # 网格区域使用 gridspec，底部留出 slider 的空间
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(
            nrows=thumbs_per_col,
            ncols=thumbs_per_row,
            figure=fig,
            bottom=0.15,  # 底下留出 slider 区域
            top=0.95,
            left=0.05,
            right=0.95,
            hspace=0.4,
            wspace=0.3,
        )

        axes_grid: List[plt.Axes] = []
        for r in range(thumbs_per_col):
            for c in range(thumbs_per_row):
                ax = fig.add_subplot(gs[r, c])
                ax.axis("off")
                axes_grid.append(ax)

        # 底部 slider
        slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(
            ax=slider_ax,
            label="Page",
            valmin=0,
            valmax=max(num_pages - 1, 0),
            valinit=0,
            valstep=1,
        )

        # 状态
        state = {
            "page": 0,
            "selected_idx": None,
            "axes_to_scene": {},  # ax -> global scene_idx
        }

        title_text = fig.suptitle(
            "Scene selection (click a thumbnail; use slider to change page)",
            fontsize=11,
        )

        def update_page(page_idx: int):
            """根据 page_idx 更新当前页显示的缩略图。"""
            page_idx = int(page_idx)
            state["page"] = page_idx
            state["axes_to_scene"].clear()

            start = page_idx * thumbs_per_page
            end = min(start + thumbs_per_page, num_scenes)

            for i, ax in enumerate(axes_grid):
                global_idx = start + i
                ax.clear()
                ax.axis("off")

                if global_idx < end:
                    prev = previews[global_idx]
                    img = prev["image"]
                    scene_name = prev["scene_name"]
                    scene_idx_global = prev["scene_idx"]

                    ax.imshow(img)
                    # 显示场景名（过长的话截断）
                    short_name = scene_name
                    if len(short_name) > 25:
                        short_name = short_name[:22] + "..."
                    ax.set_title(
                        f"{scene_idx_global}: {short_name}",
                        fontsize=8,
                        pad=4,
                    )
                    ax.axis("off")

                    state["axes_to_scene"][ax] = scene_idx_global
                else:
                    # 该格子没有对应场景，保持空白
                    pass

            title_text.set_text(
                f"Scene selection (page {page_idx+1}/{num_pages}) "
                f"- click a thumbnail to select"
            )
            fig.canvas.draw_idle()

        # 初始化第一页
        update_page(0)

        def on_slider_change(val):
            update_page(int(val))

        slider.on_changed(on_slider_change)

        def on_click(event):
            if event.inaxes is None:
                return
            ax = event.inaxes
            if ax in state["axes_to_scene"]:
                scene_idx_global = state["axes_to_scene"][ax]
                state["selected_idx"] = scene_idx_global
                if self.verbose:
                    print(f"[Bridge] Selected scene index: {scene_idx_global}")
                plt.close(fig)

        def on_key(event):
            # 允许 q / Esc 取消
            if event.key in ("q", "escape"):
                state["selected_idx"] = None
                if self.verbose:
                    print("[Bridge] Scene selection cancelled by user.")
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

        plt.show()  # 阻塞直到窗口关闭

        if state["selected_idx"] is None:
            raise RuntimeError(
                "No scene was selected. "
                "Please run again and click a thumbnail to select a scene."
            )

        return int(state["selected_idx"])



    def interactive_select_ego_and_generate_rds_hq(
        self,
        scene_idx: int,
        time_start: float,
        time_end: float,
        output_dir: Union[str, Path],
        snapshot_time: Optional[float] = None,
        streetview_retrieval: bool = False,
        agent_clip_distance: float = 80.0,
        map_clip_distance: float = 100.0,
        camera_setting: str = "default",
    ) -> Path:
        """
        完整交互流程（方案 A 已集成）：

        1) 使用 InterHub 数据在 snapshot_time 生成 BEV；
        2) 使用 InterHub 数据在 [time_start, time_end] 生成场景 GIF（带路网 + 矩形车辆）；
        3) 弹出 GUI：左侧 BEV（路网 + 矩形车）可点击选车，右侧 GIF 对照；
        4) 选中 ego 车辆后，自动计算该车在整个场景内的存在时间区间，
           将 RDS-HQ 的时间窗口收缩到：
               [max(time_start, ego_first_t), min(time_end, ego_last_t)]
           如与原窗口重叠太小，则退化为车辆自己的完整生命周期 [ego_first_t, ego_last_t]；
        5) 用调整后的时间窗口和所选 ego 调用 generate_rds_hq_from_scene() 生成 RDS-HQ。
        """
        if scene_idx < 0 or scene_idx >= self.num_scenes:
            raise ValueError(
                f"Scene index {scene_idx} out of bounds. "
                f"Environment has {self.num_scenes} scenes (0 to {self.num_scenes-1})"
            )

        # 1) 从 InterHub 缓存中加载场景
        scene_tag = self.scenes_list[scene_idx]
        scene_name = getattr(scene_tag, "name", f"scene_{scene_idx}")
        scene: Scene = self.env_cache.load_scene(
            self.env_name, scene_name, scene_dt=self.scene_dt
        )

        # 选择用于 BEV 的 snapshot 时间点
        if snapshot_time is None:
            snapshot_time = 0.5 * (time_start + time_end)

        if self.verbose:
            print(
                f"[Bridge] Building BEV for scene '{scene.name}' at t={snapshot_time:.2f}s "
                f"to select ego vehicle interactively."
            )

        # 2) 先用 snapshot_time 生成 BEV（所有车辆中心点 + 可视边界）
        positions, bounds, actual_t = self._build_bev_for_scene(
            scene=scene,
            snapshot_time=snapshot_time,
            vehicles_only=True,
        )

        if self.verbose and abs(actual_t - snapshot_time) > 1e-3:
            print(
                f"[Bridge] Snapshot time clamped to {actual_t:.2f}s "
                f"(requested {snapshot_time:.2f}s)"
            )

        # 3) 准备矢量路网（用于 BEV & GIF 背景）
        vec_map = self._get_vec_map_for_scene(scene)

        # 4) 用 [time_start, time_end] 生成场景 GIF（带路网轮廓 + 矩形车辆）
        gif_output_dir = Path(output_dir) / "interhub_gif"
        gif_path = self._generate_scene_gif(
            scene=scene,
            time_start=time_start,
            time_end=time_end,
            bounds=bounds,
            output_dir=gif_output_dir,
            max_frames=60,
            fps=5,
            vec_map=vec_map,
        )

        # 5) 弹出 GUI：左 BEV + 右 GIF，对齐边界；点击 BEV 中的车选 ego
        ego_agent_id = self._interactive_select_vehicle_from_bev(
            positions=positions,
            bounds=bounds,
            gif_path=gif_path,
            scene=scene,
            snapshot_time=actual_t,
            vec_map=vec_map,
            window_title=f"Scene {scene.name} - Select Ego",
        )

        if self.verbose:
            print(f"[Bridge] User selected ego agent: {ego_agent_id}")

        # 6) 方案 A：根据 ego 的“存在时间”自动调整 RDS-HQ 时间窗口
        dt = scene.dt
        # 找出这辆车在哪些 timestep 出现
        present_indices: List[int] = []
        for t_idx in range(scene.length_timesteps):
            for agent_meta in scene.agent_presence[t_idx]:
                if agent_meta.name == ego_agent_id:
                    present_indices.append(t_idx)
                    break

        if not present_indices:
            # 理论上不应该发生，因为我们刚刚在 snapshot_time 看见了它
            raise ValueError(
                f"Ego agent '{ego_agent_id}' not found in scene '{scene.name}' "
                "when computing presence interval."
            )

        ego_first_idx = min(present_indices)
        ego_last_idx = max(present_indices)

        # 这里为了保证在 FCD 离散时 ego 在所有帧都存在：
        # FCD 会使用 [start_idx, end_idx) 这样的整数索引区间
        # 所以我们把 ego 的时间区间设为 [ego_first_t, ego_last_t]
        # 其中 ego_last_t = (ego_last_idx + 1) * dt，保证最后一个存在帧也被覆盖
        ego_first_t = ego_first_idx * dt
        ego_last_t = (ego_last_idx + 1) * dt

        # 和用户给的 [time_start, time_end] 取交集
        adj_start = max(time_start, ego_first_t)
        adj_end = min(time_end, ego_last_t)

        # 如果交集太小（比如几乎没有重叠），退而求其次：直接用 ego 的完整生命周期
        min_duration = dt * 1.5  # 至少要有一个多 timestep 的长度
        if adj_end - adj_start < min_duration:
            if self.verbose:
                print(
                    "[Bridge] WARNING: Selected ego is not present for most of the "
                    f"requested window [{time_start:.2f}, {time_end:.2f}]s.\n"
                    f"  Ego lifetime: [{ego_first_t:.2f}, {ego_last_t:.2f}]s\n"
                    "  Using ego full lifetime as RDS-HQ window."
                )
            adj_start = ego_first_t
            adj_end = ego_last_t
        else:
            if self.verbose:
                print(
                    f"[Bridge] Adjusted RDS-HQ time window to ego presence intersection:\n"
                    f"  original: [{time_start:.2f}, {time_end:.2f}]s\n"
                    f"  ego life: [{ego_first_t:.2f}, {ego_last_t:.2f}]s\n"
                    f"  final:    [{adj_start:.2f}, {adj_end:.2f}]s"
                )

        # 防御性：再剪一次到场景合法范围
        scene_duration = scene.length_timesteps * dt
        adj_start = max(0.0, min(adj_start, scene_duration))
        adj_end = max(0.0, min(adj_end, scene_duration))
        if adj_end <= adj_start + 1e-6:
            raise ValueError(
                f"After adjustment, invalid time window for ego '{ego_agent_id}': "
                f"[{adj_start:.2f}, {adj_end:.2f}]s. "
                "Please try a different ego vehicle or time range."
            )

        if self.verbose:
            print(
                f"[Bridge] Final RDS-HQ time window for ego '{ego_agent_id}': "
                f"[{adj_start:.2f}, {adj_end:.2f}]s"
            )

        # 7) 用调整后的时间窗口 + 选定的 ego 进入原来的 RDS-HQ 生成管线
        return self.generate_rds_hq_from_scene(
            scene_idx=scene_idx,
            time_start=adj_start,
            time_end=adj_end,
            output_dir=output_dir,
            ego_agent_id=ego_agent_id,
            streetview_retrieval=streetview_retrieval,
            agent_clip_distance=agent_clip_distance,
            map_clip_distance=map_clip_distance,
            camera_setting=camera_setting,
        )



    # ------------------------------------------------------------------
    # 其他工具方法
    # ------------------------------------------------------------------
    def batch_process_interactions(
        self,
        interaction_csv_path: str,
        output_base_dir: str,
        max_scenes: Optional[int] = None,
        streetview_retrieval: bool = False,
        agent_clip_distance: float = 80.0,
        camera_setting: str = "default",
    ) -> List[Path]:
        """
        Process multiple interaction scenarios from InterHub's extraction.

        Args:
            interaction_csv_path: Path to InterHub's interaction results CSV
            output_base_dir: Base directory for all outputs
            max_scenes: Maximum number to process (None = all)
            streetview_retrieval: Whether to fetch street view images
            agent_clip_distance: Agent clipping distance
            camera_setting: Camera configuration preset

        Returns:
            List of paths to generated RDS-HQ directories
        """
        df = pd.read_csv(interaction_csv_path)
        if max_scenes is not None:
            df = df.head(max_scenes)

        print(f"\nProcessing {len(df)} interaction scenarios...")
        output_dirs: List[Path] = []

        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc="Processing interactions"
        ):
            try:
                scene_idx = int(row.get("scene_idx", idx))
                time_start = float(row.get("time_start", 0.0))
                time_end = float(row.get("time_end", 10.0))

                output_dir = Path(output_base_dir) / f"interaction_{idx:04d}"
                rds_path = self.generate_rds_hq_from_scene(
                    scene_idx=scene_idx,
                    time_start=time_start,
                    time_end=time_end,
                    output_dir=str(output_dir),
                    streetview_retrieval=streetview_retrieval,
                    agent_clip_distance=agent_clip_distance,
                    camera_setting=camera_setting,
                )
                output_dirs.append(rds_path)

            except Exception as e:
                print(f"\nError processing interaction {idx}: {e}")
                continue

        print(f"\n✓ Batch processing complete: {len(output_dirs)} scenarios generated")
        return output_dirs

    def get_scene_info(self, scene_idx: int) -> Dict:
        """
        Get information about a specific scene.

        Args:
            scene_idx: Scene index

        Returns:
            Dictionary with scene information
        """
        if scene_idx < 0 or scene_idx >= self.num_scenes:
            raise ValueError(
                f"Scene index {scene_idx} out of range [0, {self.num_scenes})"
            )

        scene_tag = self.scenes_list[scene_idx]
        scene_name = getattr(scene_tag, "name", f"scene_{scene_idx}")
        scene = self.env_cache.load_scene(
            self.env_name, scene_name, scene_dt=self.scene_dt
        )

        # Count agents by type
        agent_type_counts: Dict[str, int] = {}
        all_agent_names = set()

        for t_idx in range(scene.length_timesteps):
            for agent_meta in scene.agent_presence[t_idx]:
                agent_name = agent_meta.name
                all_agent_names.add(agent_name)

                # Get agent type from scene.agents dictionary
                if agent_name in scene.agents:
                    agent_info = scene.agents[agent_name]
                    if hasattr(agent_info, "agent_type"):
                        agent_type = str(agent_info.agent_type)
                    else:
                        agent_type = "UNKNOWN"
                else:
                    agent_type = "UNKNOWN"

                agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

        return {
            "name": scene.name,
            "duration_seconds": scene.length_timesteps * scene.dt,
            "num_timesteps": scene.length_timesteps,
            "dt": scene.dt,
            "num_agents": len(all_agent_names),
            "agent_type_counts": agent_type_counts,
        }


# ----------------------------------------------------------------------
# 便捷封装函数
# ----------------------------------------------------------------------
def convert_interhub_scene_to_rds_hq(
    cache_path: Union[str, Path],
    scene_idx: int,
    time_start: float,
    time_end: float,
    output_dir: Union[str, Path],
    dataset_name: str = "interaction_multi",
    ego_agent_id: Optional[str] = None,
    streetview_retrieval: bool = False,
    agent_clip_distance: float = 80.0,
    camera_setting: str = "default",
) -> Path:
    """
    非交互版本：直接转换为 RDS-HQ。
    """
    bridge = InterHubToTeraSimBridge(
        interhub_cache_path=cache_path,
        dataset_name=dataset_name,
        verbose=True,
    )

    return bridge.generate_rds_hq_from_scene(
        scene_idx=scene_idx,
        time_start=time_start,
        time_end=time_end,
        output_dir=output_dir,
        ego_agent_id=ego_agent_id,
        streetview_retrieval=streetview_retrieval,
        agent_clip_distance=agent_clip_distance,
        camera_setting=camera_setting,
    )


from typing import Optional  # 顶部如果还没有 Optional 记得 import 一下

def interactive_convert_interhub_scene_to_rds_hq(
    cache_path: Union[str, Path],
    scene_idx: Optional[int],
    time_start: float,
    time_end: float,
    output_dir: Union[str, Path],
    dataset_name: str = "interaction_multi",
    snapshot_time: Optional[float] = None,
    streetview_retrieval: bool = False,
    agent_clip_distance: float = 80.0,
    map_clip_distance: float = 100.0,
    camera_setting: str = "default",
) -> Path:
    """
    交互版本（扩展）：

    1. 如果 scene_idx 为 None，则先弹出“场景选择 + GIF 预览”GUI，
       让用户在本地所有 InterHub 场景中选择一个；
    2. 然后对选中的场景再弹出 BEV GUI，点击选中 ego 车辆（高亮）；
    3. 最后使用“方案 A 调整后的时间窗口” + 选中的 ego 生成 RDS-HQ。
    """
    bridge = InterHubToTeraSimBridge(
        interhub_cache_path=cache_path,
        dataset_name=dataset_name,
        verbose=True,
    )

    # ✅ 关键：如果 scene_idx 为 None，先让用户在 GUI 里选择一个场景
    if scene_idx is None:
        scene_idx = bridge.interactive_select_scene_with_gif(
            time_start=time_start,
            time_end=time_end,
            preview_output_dir=output_dir,
            max_frames=60,
            fps=5,
        )

    # 到这里 scene_idx 一定是 int，不会再是 None
    return bridge.interactive_select_ego_and_generate_rds_hq(
        scene_idx=scene_idx,
        time_start=time_start,
        time_end=time_end,
        output_dir=output_dir,
        snapshot_time=snapshot_time,
        streetview_retrieval=streetview_retrieval,
        agent_clip_distance=agent_clip_distance,
        map_clip_distance=map_clip_distance,
        camera_setting=camera_setting,
    )
