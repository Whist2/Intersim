"""
InterHub-TeraSim Direct Integration Bridge (Cosmos-Drive-Dreams Enhanced Version)

在 FIXED VERSION 的基础上增加：

1. 使用 InterHub 轨迹数据丰富 SUMO 路网：
   - 保留原有规则栅格网络，保证稳健
   - 额外从 FCD 中抽取真实车辆轨迹，为位移足够大的车辆生成独立 edge/lane
   - lane.shape 使用实际轨迹折线，使 HDMap 更贴近真实道路几何

2. 为 Cosmos-Drive-Dreams 输入标准准备统一颜色配置：
   - 默认提供 Cosmos 风格的 HDMap 颜色与类型配置
   - 支持通过 hdmap_color_config / hdmap_type_config 自定义
   - 在传给 TeraSimToCosmosConverter 的 config_dict 中注入颜色与视频配置，
     方便在 TeraSim 端用 Cosmos-Drive-Dreams 标准渲染视频
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
    # 注意：这里的颜色键仅作为配置约定，具体如何使用由 TeraSim 端决定。
    DEFAULT_COSMOS_HD_MAP_COLOR_CONFIG: Dict[str, List[int]] = {
        # 背景（非道路）
        "background": [0, 0, 0],
        # 可行驶区域 / drivable area
        "drivable_area": [40, 40, 40],
        # 车道中心线 / lane center (虚线/实线)
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

    # HDMap 类型到语义 id 的映射（可被下游用作语义通道）
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
            fcd_path=fcd_path,  # 利用已经生成的 FCD 进一步丰富路网
        )

        if self.verbose:
            print(f"✓ Map file created: {map_path}")

        # ===== STEP 4: Run TeraSim RDS-HQ Converter =====
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 4/4: Running TeraSim RDS-HQ Converter")
            print("=" * 70)

        # 推导与 Cosmos-Drive-Dreams 对齐的视频配置（30 FPS, 704×1280）
        # 注意：具体是否严格使用这些配置由 TeraSimToCosmosConverter 决定。
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
            # 新增：Cosmos-Drive-Dreams 相关配置（由 TeraSim 侧解释使用）
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

                # Extract heading
                heading_val = float(raw_state[heading_idx])
                angle_deg = float(np.degrees(heading_val))

                # Calculate speed from velocity
                speed = 0.0
                if vx_idx is not None and vy_idx is not None:
                    vx = float(raw_state[vx_idx])
                    vy = float(raw_state[vy_idx])
                    speed = float(np.sqrt(vx**2 + vy**2))

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
        import pandas as pd

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


# Convenience function
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
    Convenience function for one-off conversions.

    Args:
        cache_path: Path to InterHub unified cache
        scene_idx: Scene index to convert
        time_start: Start time in seconds
        time_end: End time in seconds
        output_dir: Output directory
        dataset_name: Dataset name in cache
        ego_agent_id: Optional ego vehicle ID
        streetview_retrieval: Enable street view retrieval
        agent_clip_distance: Agent clipping distance
        camera_setting: Camera configuration

    Returns:
        Path to generated RDS-HQ directory
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
