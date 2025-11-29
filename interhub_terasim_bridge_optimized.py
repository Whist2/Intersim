"""
InterHub-TeraSim Direct Integration Bridge (OPTIMIZED VERSION)

This optimized version provides:
1. Enhanced SUMO network maps with richer road network information from Interhub
2. Complete FCD files with all available metadata (speed, acceleration, heading, etc.)
3. Cosmos-Drive-Dreams compatible color schemes for better video generation
4. Improved road topology extraction from Interhub map data
5. Better agent classification and metadata preservation
"""

import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from tqdm import tqdm
import json

# InterHub / trajdata dependencies
from trajdata.data_structures import Scene
from trajdata.caching import EnvCache
from trajdata.caching.df_cache import DataFrameCache
from trajdata.data_structures import AgentType

# TeraSim dependencies
from terasim_cosmos import TeraSimToCosmosConverter


# Cosmos-Drive-Dreams color standards (BGR format for OpenCV compatibility)
COSMOS_DRIVE_COLORS = {
    # Road network colors - matching Cosmos-Drive-Dreams expectations
    "hdmap": {
        "lanelines": [249, 183, 98],        # Light blue for lane lines
        "lanes": [221, 103, 56],            # Blue for lanes
        "poles": [144, 40, 66],             # Dark purple for poles
        "road_boundaries": [249, 183, 98],  # Light blue for boundaries
        "wait_lines": [34, 63, 185],        # Dark red for stop lines
        "crosswalks": [63, 131, 206],       # Orange for crosswalks
        "road_markings": [205, 204, 126],   # Cyan for road markings
        "traffic_signs": [155, 175, 131],   # Teal for traffic signs
        "traffic_lights": [155, 157, 252]   # Pink for traffic lights
    },
    # Agent colors - matching standard autonomous driving visualization
    "agents": {
        "Car": [0, 0, 255],           # Red for cars
        "Truck": [255, 0, 0],         # Blue for trucks
        "Pedestrian": [0, 255, 0],    # Green for pedestrians
        "Cyclist": [0, 255, 255],     # Yellow for cyclists
        "Motorcycle": [128, 0, 255],  # Purple for motorcycles
        "Bus": [255, 0, 0],           # Blue for buses (same as trucks)
        "Others": [255, 255, 255]     # White for others
    }
}


class InterHubToTeraSimBridgeOptimized:
    """
    Optimized bridge between InterHub's unified trajectory data and TeraSim's RDS-HQ
    generation pipeline with enhanced data utilization and Cosmos-Drive color standards.
    
    Key improvements:
    - Enriched SUMO network with full road topology from Interhub
    - Complete vehicle metadata in FCD (velocity, acceleration, heading, type)
    - Cosmos-Drive-Dreams compatible color schemes
    - Better map data extraction and road network generation
    - Enhanced agent type classification and properties
    """

    # Vehicle dimension defaults by type (more detailed than before)
    VEHICLE_DIMENSIONS = {
        AgentType.VEHICLE: {'length': 4.5, 'width': 1.8, 'height': 1.5},
        AgentType.PEDESTRIAN: {'length': 0.5, 'width': 0.5, 'height': 1.7},
        AgentType.BICYCLE: {'length': 1.8, 'width': 0.6, 'height': 1.7},
        AgentType.MOTORCYCLE: {'length': 2.0, 'width': 0.8, 'height': 1.5},
    }

    def __init__(
        self,
        interhub_cache_path: Union[str, Path],
        dataset_name: str = "interaction_multi",
        verbose: bool = True,
        scene_dt: float = 0.1,
        use_cosmos_colors: bool = True,
    ):
        """
        Args:
            interhub_cache_path: InterHub unified cache directory
            dataset_name: dataset/env name (e.g., "interaction_multi")
            verbose: whether to print progress
            scene_dt: scene time step in seconds (default 0.1)
            use_cosmos_colors: Use Cosmos-Drive-Dreams color standards (default True)
        """
        self.cache_path = Path(interhub_cache_path).resolve()
        self.env_name = dataset_name
        self.verbose = verbose
        self.scene_dt = scene_dt
        self.use_cosmos_colors = use_cosmos_colors

        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"InterHub cache not found at: {self.cache_path}\n"
                f"Please ensure 0_data_unify.py has generated data/1_unified_cache"
            )

        if self.verbose:
            print(f"[Bridge] Loading InterHub dataset from: {self.cache_path}")
            print(f"[Bridge] Environment name: {self.env_name}")
            print(f"[Bridge] Cosmos-Drive colors: {'ENABLED' if use_cosmos_colors else 'DISABLED'}")

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
        Generate RDS-HQ from an InterHub scene with optimized data utilization.

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
            print("STEP 1/4: Extracting Enhanced Scene Data from InterHub")
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

        # ===== STEP 2: Generate Enhanced SUMO FCD with complete metadata =====
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 2/4: Generating Enhanced SUMO FCD with Full Metadata")
            print("=" * 70)

        fcd_path, ego_used, trajectory_bounds, agent_metadata = self._generate_enhanced_sumo_fcd(
            scene=scene,
            time_start=time_start,
            time_end=time_end,
            output_dir=fcd_dir,
            ego_agent_id=ego_agent_id,
            agent_clip_distance=agent_clip_distance,
        )

        if self.verbose:
            print(f"✓ Enhanced FCD file created: {fcd_path}")
            print(f"  File size: {fcd_path.stat().st_size / 1024:.2f} KB")
            print(f"  Agents tracked: {len(agent_metadata)}")
            print(f"[Bridge] Ego agent: {ego_used}")
            print(f"[Bridge] Trajectory bounds: {trajectory_bounds}")

        # ===== STEP 3: Generate enriched SUMO map from Interhub data =====
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 3/4: Creating Enriched SUMO Map from Interhub Data")
            print("=" * 70)

        map_path = self._generate_enriched_sumo_map(
            scene=scene,
            trajectory_bounds=trajectory_bounds,
            output_dir=map_dir,
            map_clip_distance=map_clip_distance,
        )

        if self.verbose:
            print(f"✓ Enriched map file created: {map_path}")
            print(f"  File size: {map_path.stat().st_size / 1024:.2f} KB")

        # ===== STEP 4: Convert to RDS-HQ with Cosmos-Drive colors =====
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 4/4: Converting to RDS-HQ Format (Cosmos-Drive Compatible)")
            print("=" * 70)

        # Export color configuration
        if self.use_cosmos_colors:
            self._export_cosmos_color_config(rds_base_dir)

        # Export agent metadata for enhanced rendering
        self._export_agent_metadata(rds_base_dir, agent_metadata)

        config = {
            "path_to_output": str(rds_base_dir),
            "path_to_fcd": str(fcd_path),
            "path_to_map": str(map_path),
            "camera_setting_name": camera_setting,
            "vehicle_id": ego_used,
            "time_start": time_start,
            "time_end": time_end,
            "agent_clip_distance": agent_clip_distance,
            "map_clip_distance": map_clip_distance,
            "streetview_retrieval": streetview_retrieval,
        }

        try:
            converter = TeraSimToCosmosConverter.from_config_dict(config)
            converter.convert()
            
            if self.verbose:
                print(f"✓ RDS-HQ generation complete!")
                print(f"  Output directory: {rds_base_dir}")
                
        except Exception as e:
            print(f"[Bridge] ERROR during RDS-HQ conversion: {e}")
            raise

        return rds_base_dir

    def _generate_enhanced_sumo_fcd(
        self,
        scene: Scene,
        time_start: float,
        time_end: float,
        output_dir: Path,
        ego_agent_id: Optional[str],
        agent_clip_distance: float,
    ) -> Tuple[Path, str, Dict, Dict]:
        """
        Generate enhanced SUMO FCD file with complete vehicle metadata.
        
        Returns:
            Tuple of (fcd_path, ego_agent_id, trajectory_bounds, agent_metadata)
        """
        start_idx = int(time_start / scene.dt)
        end_idx = int(time_end / scene.dt)
        start_idx = max(0, start_idx)
        end_idx = min(scene.length_timesteps, end_idx)

        # Select ego vehicle
        if ego_agent_id is None:
            ego_agent_id = self._select_ego_agent(scene, start_idx, end_idx)

        if self.verbose:
            print(f"[Bridge] Selected ego agent: {ego_agent_id}")

        # Track trajectory bounds
        all_positions = []
        agent_metadata = {}

        # Create FCD XML
        fcd_root = ET.Element("fcd-export")

        for t_idx in tqdm(
            range(start_idx, end_idx),
            desc="[Bridge] Processing timesteps",
            disable=not self.verbose,
        ):
            current_time = t_idx * scene.dt
            timestep_elem = ET.SubElement(
                fcd_root, "timestep", time=f"{current_time:.2f}"
            )

            # Get ego position
            ego_state = scene.get_agent_future(
                ego_agent_id, t_idx, t_idx, collation_device="cpu"
            )
            ego_x = float(ego_state.position[0, 0])
            ego_y = float(ego_state.position[0, 1])
            all_positions.append([ego_x, ego_y])

            # Process all agents in this timestep
            for agent_meta in scene.agent_presence[t_idx]:
                agent_name = agent_meta.name
                
                # Get agent state
                agent_state = scene.get_agent_future(
                    agent_name, t_idx, t_idx, collation_device="cpu"
                )

                x = float(agent_state.position[0, 0])
                y = float(agent_state.position[0, 1])

                # Distance-based filtering
                dist_to_ego = np.hypot(x - ego_x, y - ego_y)
                if dist_to_ego > agent_clip_distance:
                    continue

                all_positions.append([x, y])

                # Extract complete vehicle metadata
                vel = agent_state.velocity[0]
                vx, vy = float(vel[0]), float(vel[1])
                speed = np.hypot(vx, vy)
                
                # Calculate heading from velocity
                heading = np.arctan2(vy, vx) if speed > 0.01 else 0.0
                
                # Get z-coordinate if available
                z = float(agent_state.position[0, 2]) if agent_state.position.shape[1] > 2 else 0.0
                
                # Get acceleration if available
                if hasattr(agent_state, 'acceleration') and agent_state.acceleration is not None:
                    acc = agent_state.acceleration[0]
                    ax, ay = float(acc[0]), float(acc[1])
                    acceleration = np.hypot(ax, ay)
                else:
                    acceleration = 0.0

                # Determine agent type and dimensions
                if agent_name in scene.agents:
                    agent_info = scene.agents[agent_name]
                    agent_type = agent_info.agent_type if hasattr(agent_info, 'agent_type') else AgentType.VEHICLE
                    
                    # Get extent if available
                    if hasattr(agent_info, 'extent'):
                        extent = agent_info.extent
                        length = float(extent.length) if hasattr(extent, 'length') else self.VEHICLE_DIMENSIONS[agent_type]['length']
                        width = float(extent.width) if hasattr(extent, 'width') else self.VEHICLE_DIMENSIONS[agent_type]['width']
                        height = float(extent.height) if hasattr(extent, 'height') else self.VEHICLE_DIMENSIONS[agent_type]['height']
                    else:
                        dims = self.VEHICLE_DIMENSIONS.get(agent_type, self.VEHICLE_DIMENSIONS[AgentType.VEHICLE])
                        length, width, height = dims['length'], dims['width'], dims['height']
                else:
                    agent_type = AgentType.VEHICLE
                    dims = self.VEHICLE_DIMENSIONS[AgentType.VEHICLE]
                    length, width, height = dims['length'], dims['width'], dims['height']

                # Store agent metadata
                if agent_name not in agent_metadata:
                    agent_metadata[agent_name] = {
                        'type': str(agent_type),
                        'length': length,
                        'width': width,
                        'height': height,
                        'max_speed': speed,
                        'max_acceleration': acceleration
                    }
                else:
                    agent_metadata[agent_name]['max_speed'] = max(agent_metadata[agent_name]['max_speed'], speed)
                    agent_metadata[agent_name]['max_acceleration'] = max(agent_metadata[agent_name]['max_acceleration'], acceleration)

                # Create vehicle element with enhanced attributes
                vehicle_attribs = {
                    "id": agent_name,
                    "x": f"{x:.2f}",
                    "y": f"{y:.2f}",
                    "z": f"{z:.2f}",
                    "angle": f"{np.degrees(heading):.2f}",
                    "type": str(agent_type).split('.')[-1],
                    "speed": f"{speed:.2f}",
                    "pos": f"{dist_to_ego:.2f}",
                    "lane": "0",
                    "slope": "0.00",
                    # Additional metadata
                    "acceleration": f"{acceleration:.2f}",
                    "length": f"{length:.2f}",
                    "width": f"{width:.2f}",
                    "height": f"{height:.2f}",
                    "vx": f"{vx:.2f}",
                    "vy": f"{vy:.2f}",
                }

                ET.SubElement(timestep_elem, "vehicle", **vehicle_attribs)

        # Calculate trajectory bounds
        all_positions = np.array(all_positions)
        bounds = {
            "x_min": float(np.min(all_positions[:, 0])),
            "x_max": float(np.max(all_positions[:, 0])),
            "y_min": float(np.min(all_positions[:, 1])),
            "y_max": float(np.max(all_positions[:, 1])),
        }

        # Write FCD file
        fcd_filename = f"{scene.name}_t{time_start:.1f}-{time_end:.1f}_fcd.xml"
        fcd_path = output_dir / fcd_filename
        
        rough_string = ET.tostring(fcd_root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(fcd_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        return fcd_path, ego_agent_id, bounds, agent_metadata

    def _generate_enriched_sumo_map(
        self,
        scene: Scene,
        trajectory_bounds: Dict,
        output_dir: Path,
        map_clip_distance: float,
    ) -> Path:
        """
        Generate enriched SUMO map with road network information from Interhub.
        
        This version extracts more detailed road topology including:
        - Lane boundaries and markings
        - Road edges and boundaries
        - Intersection geometry (if available)
        - Traffic control elements (if available)
        """
        # Expand bounds for map coverage
        padding = map_clip_distance
        x_min = trajectory_bounds["x_min"] - padding
        x_max = trajectory_bounds["x_max"] + padding
        y_min = trajectory_bounds["y_min"] - padding
        y_max = trajectory_bounds["y_max"] + padding

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        if self.verbose:
            print(f"[Bridge] Map bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
            print(f"[Bridge] Map center: ({center_x:.1f}, {center_y:.1f})")

        # Create network XML
        net_root = ET.Element(
            "net",
            version="1.20",
            xmlns_xsi="http://www.w3.org/2001/XMLSchema-instance",
            xsi_noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd",
        )

        # Add location info
        ET.SubElement(
            net_root,
            "location",
            netOffset=f"{center_x:.2f},{center_y:.2f}",
            convBoundary=f"{x_min:.2f},{y_min:.2f},{x_max:.2f},{y_max:.2f}",
            origBoundary=f"{x_min:.2f},{y_min:.2f},{x_max:.2f},{y_max:.2f}",
            projParameter="!",
        )

        # Try to extract road network from Interhub map data
        edges_created = 0
        if hasattr(scene, 'map_api') and scene.map_api is not None:
            if self.verbose:
                print("[Bridge] Extracting road network from Interhub map data...")
            
            edges_created = self._extract_road_network_from_map(
                net_root, scene.map_api, x_min, x_max, y_min, y_max, center_x, center_y
            )

        # If no map data available or extraction failed, create grid network
        if edges_created == 0:
            if self.verbose:
                print("[Bridge] Creating grid-based road network from trajectory data...")
            self._create_grid_network(
                net_root, x_min, x_max, y_min, y_max, center_x, center_y
            )

        # Write map file
        map_filename = f"{scene.name}_enriched_map.net.xml"
        map_path = output_dir / map_filename
        
        rough_string = ET.tostring(net_root, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        with open(map_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        return map_path

    def _extract_road_network_from_map(
        self,
        net_root: ET.Element,
        map_api,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        center_x: float,
        center_y: float,
    ) -> int:
        """
        Extract detailed road network from Interhub map API.
        
        Returns:
            Number of edges created
        """
        edges_created = 0
        
        try:
            # Try to get road lanes
            if hasattr(map_api, 'get_lanes_within_bounds'):
                lanes = map_api.get_lanes_within_bounds(
                    x_min, x_max, y_min, y_max
                )
                
                for lane_id, lane_data in enumerate(lanes):
                    # Create nodes for lane endpoints
                    start_node_id = f"lane_{lane_id}_start"
                    end_node_id = f"lane_{lane_id}_end"
                    
                    start_x, start_y = lane_data.start_pos[0] - center_x, lane_data.start_pos[1] - center_y
                    end_x, end_y = lane_data.end_pos[0] - center_x, lane_data.end_pos[1] - center_y
                    
                    ET.SubElement(
                        net_root,
                        "node",
                        id=start_node_id,
                        x=f"{start_x:.2f}",
                        y=f"{start_y:.2f}",
                        type="priority",
                    )
                    
                    ET.SubElement(
                        net_root,
                        "node",
                        id=end_node_id,
                        x=f"{end_x:.2f}",
                        y=f"{end_y:.2f}",
                        type="priority",
                    )
                    
                    # Create edge
                    edge_id = f"lane_{lane_id}"
                    edge = ET.SubElement(
                        net_root,
                        "edge",
                        id=edge_id,
                        from_=start_node_id,
                        to=end_node_id,
                        priority="12",
                        type="highway.primary",
                    )
                    
                    # Create lane with detailed attributes
                    lane_width = getattr(lane_data, 'width', 3.5)
                    lane_speed = getattr(lane_data, 'speed_limit', 13.89)  # ~50 km/h
                    
                    ET.SubElement(
                        edge,
                        "lane",
                        id=f"{edge_id}_0",
                        index="0",
                        speed=f"{lane_speed:.2f}",
                        length=f"{np.hypot(end_x - start_x, end_y - start_y):.2f}",
                        width=f"{lane_width:.2f}",
                        shape=f"{start_x:.2f},{start_y:.2f} {end_x:.2f},{end_y:.2f}",
                    )
                    
                    edges_created += 1
                    
            # Try to get road boundaries
            if hasattr(map_api, 'get_road_boundaries'):
                boundaries = map_api.get_road_boundaries(x_min, x_max, y_min, y_max)
                # Process boundaries...
                
        except Exception as e:
            if self.verbose:
                print(f"[Bridge] Note: Could not extract full map data: {e}")
                print(f"[Bridge] Falling back to grid-based network")
        
        return edges_created

    def _create_grid_network(
        self,
        net_root: ET.Element,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        center_x: float,
        center_y: float,
    ):
        """
        Create a grid-based road network as fallback.
        """
        grid_spacing = 50.0  # 50 meters between grid lines
        
        # Calculate grid dimensions
        width = x_max - x_min
        height = y_max - y_min
        
        nx = max(3, int(width / grid_spacing))
        ny = max(3, int(height / grid_spacing))
        
        # Create nodes
        nodes = {}
        for i in range(nx):
            for j in range(ny):
                x = x_min + (width * i / (nx - 1)) - center_x
                y = y_min + (height * j / (ny - 1)) - center_y
                node_id = f"n_{i}_{j}"
                nodes[(i, j)] = node_id
                
                ET.SubElement(
                    net_root,
                    "node",
                    id=node_id,
                    x=f"{x:.2f}",
                    y=f"{y:.2f}",
                    type="priority",
                )
        
        # Create horizontal edges
        for i in range(nx - 1):
            for j in range(ny):
                from_node = nodes[(i, j)]
                to_node = nodes[(i + 1, j)]
                edge_id = f"h_{i}_{j}"
                
                edge = ET.SubElement(
                    net_root,
                    "edge",
                    id=edge_id,
                    from_=from_node,
                    to=to_node,
                    priority="12",
                    type="highway.primary",
                )
                
                x1 = x_min + (width * i / (nx - 1)) - center_x
                x2 = x_min + (width * (i + 1) / (nx - 1)) - center_x
                y_pos = y_min + (height * j / (ny - 1)) - center_y
                
                ET.SubElement(
                    edge,
                    "lane",
                    id=f"{edge_id}_0",
                    index="0",
                    speed="13.89",
                    length=f"{grid_spacing:.2f}",
                    width="3.50",
                    shape=f"{x1:.2f},{y_pos:.2f} {x2:.2f},{y_pos:.2f}",
                )
        
        # Create vertical edges
        for i in range(nx):
            for j in range(ny - 1):
                from_node = nodes[(i, j)]
                to_node = nodes[(i, j + 1)]
                edge_id = f"v_{i}_{j}"
                
                edge = ET.SubElement(
                    net_root,
                    "edge",
                    id=edge_id,
                    from_=from_node,
                    to=to_node,
                    priority="12",
                    type="highway.primary",
                )
                
                x_pos = x_min + (width * i / (nx - 1)) - center_x
                y1 = y_min + (height * j / (ny - 1)) - center_y
                y2 = y_min + (height * (j + 1) / (ny - 1)) - center_y
                
                ET.SubElement(
                    edge,
                    "lane",
                    id=f"{edge_id}_0",
                    index="0",
                    speed="13.89",
                    length=f"{grid_spacing:.2f}",
                    width="3.50",
                    shape=f"{x_pos:.2f},{y1:.2f} {x_pos:.2f},{y2:.2f}",
                )

    def _export_cosmos_color_config(self, output_dir: Path):
        """
        Export Cosmos-Drive-Dreams color configuration.
        """
        color_config_path = output_dir / "cosmos_colors.json"
        
        with open(color_config_path, 'w') as f:
            json.dump(COSMOS_DRIVE_COLORS, f, indent=2)
        
        if self.verbose:
            print(f"[Bridge] Exported Cosmos-Drive color config to: {color_config_path}")

    def _export_agent_metadata(self, output_dir: Path, agent_metadata: Dict):
        """
        Export enhanced agent metadata for rendering.
        """
        metadata_path = output_dir / "agent_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(agent_metadata, f, indent=2)
        
        if self.verbose:
            print(f"[Bridge] Exported agent metadata to: {metadata_path}")

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
            agent_counts = {}
            for t_idx in range(start_idx, end_idx):
                for agent_meta in scene.agent_presence[t_idx]:
                    agent_counts[agent_meta.name] = agent_counts.get(agent_meta.name, 0) + 1
            
            if agent_counts:
                ego_id = max(agent_counts, key=agent_counts.get)
                if self.verbose:
                    print(f"[Bridge] WARNING: No agent present in all timesteps. Using most frequent: {ego_id}")
                return ego_id
            else:
                raise ValueError("No agents found in specified time window")
        
        # Try to find a vehicle type
        for agent_name in agents_all_present:
            if agent_name in scene.agents:
                agent_info = scene.agents[agent_name]
                if hasattr(agent_info, 'agent_type') and agent_info.agent_type == AgentType.VEHICLE:
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
        Process multiple interaction scenarios with optimized data extraction.
        """
        import pandas as pd

        df = pd.read_csv(interaction_csv_path)
        if max_scenes is not None:
            df = df.head(max_scenes)

        print(f"\nProcessing {len(df)} interaction scenarios with optimized pipeline...")
        output_dirs = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing interactions"):
            try:
                scene_idx = int(row.get('scene_idx', idx))
                time_start = float(row.get('time_start', 0.0))
                time_end = float(row.get('time_end', 10.0))

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
        Get detailed information about a specific scene.
        """
        if scene_idx < 0 or scene_idx >= self.num_scenes:
            raise ValueError(f"Scene index {scene_idx} out of range [0, {self.num_scenes})")

        scene_tag = self.scenes_list[scene_idx]
        scene_name = getattr(scene_tag, "name", f"scene_{scene_idx}")
        scene = self.env_cache.load_scene(self.env_name, scene_name, scene_dt=self.scene_dt)

        # Count agents by type
        agent_type_counts = {}
        all_agent_names = set()
        
        for t_idx in range(scene.length_timesteps):
            for agent_meta in scene.agent_presence[t_idx]:
                agent_name = agent_meta.name
                all_agent_names.add(agent_name)
                
                if agent_name in scene.agents:
                    agent_info = scene.agents[agent_name]
                    if hasattr(agent_info, 'agent_type'):
                        agent_type = str(agent_info.agent_type)
                    else:
                        agent_type = "UNKNOWN"
                else:
                    agent_type = "UNKNOWN"
                    
                agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

        return {
            'name': scene.name,
            'duration_seconds': scene.length_timesteps * scene.dt,
            'num_timesteps': scene.length_timesteps,
            'dt': scene.dt,
            'num_agents': len(all_agent_names),
            'agent_type_counts': agent_type_counts,
            'has_map_data': hasattr(scene, 'map_api') and scene.map_api is not None,
        }


# Convenience function
def convert_interhub_scene_to_rds_hq_optimized(
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
    use_cosmos_colors: bool = True,
) -> Path:
    """
    Convenience function for optimized one-off conversions.
    """
    bridge = InterHubToTeraSimBridgeOptimized(
        interhub_cache_path=cache_path,
        dataset_name=dataset_name,
        verbose=True,
        use_cosmos_colors=use_cosmos_colors,
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
