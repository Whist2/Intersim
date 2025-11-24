"""
Optimized InterHub-TeraSim Bridge for Cosmos-Drive-Dreams (FIXED v3)
=====================================================================

FIXES v3:
- Fixed: scene.agents is a LIST, not a dict (AttributeError: 'list' has no 'keys')
- Fixed: get_agent_history -> DataFrameCache.get_raw_state()
- Fixed: Relaxed ego agent selection for real-world scenarios
- Added: Dynamic timestep adjustment based on agent presence
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from tqdm import tqdm
import cv2

from trajdata.data_structures import Scene
from trajdata.caching import EnvCache
from trajdata.data_structures import AgentType
from trajdata.maps.vec_map import VectorMap
from trajdata.caching.df_cache import DataFrameCache

try:
    from terasim_cosmos import TeraSimToCosmosConverter
except ImportError:
    pass


COSMOS_COLORS = {
    'lane_line_white': (255, 255, 255),
    'lane_line_yellow': (0, 255, 255),
    'lane_line_dashed': (200, 200, 200),
    'road_boundary': (255, 0, 0),
    'curb': (255, 100, 100),
    'crosswalk': (0, 255, 255),
    'stop_line': (255, 0, 255),
    'road_marking': (255, 255, 0),
    'pole': (128, 128, 255),
    'traffic_light': (0, 255, 0),
    'traffic_sign': (255, 165, 0),
    'vehicle': (0, 255, 0),
    'pedestrian': (255, 255, 0),
    'bicycle': (0, 255, 255),
    'motorcycle': (255, 128, 0),
    'background': (0, 0, 0),
}

COSMOS_LINE_WIDTHS = {
    'lane_line': 2,
    'road_boundary': 3,
    'crosswalk': 4,
    'stop_line': 3,
    'road_marking': 2,
    'pole': 5,
    'traffic_element': 8,
    'cuboid': 2,
}


class CosmosMapRenderer:
    """Renders HDMap elements with Cosmos-Drive-Dreams color scheme."""
    
    def __init__(self, image_width: int = 1280, image_height: int = 720):
        self.width = image_width
        self.height = image_height
        
    def create_hdmap_frame(self, map_elements: Dict, camera_params: Dict, 
                          ego_pose: np.ndarray) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self._render_road_boundaries(frame, map_elements.get('road_boundaries', []), 
                                     camera_params, ego_pose)
        self._render_lane_lines(frame, map_elements.get('lane_lines', []), 
                               camera_params, ego_pose)
        self._render_crosswalks(frame, map_elements.get('crosswalks', []), 
                               camera_params, ego_pose)
        self._render_road_markings(frame, map_elements.get('road_markings', []), 
                                   camera_params, ego_pose)
        self._render_poles(frame, map_elements.get('poles', []), 
                          camera_params, ego_pose)
        self._render_traffic_elements(frame, map_elements.get('traffic_lights', []),
                                      map_elements.get('traffic_signs', []),
                                      camera_params, ego_pose)
        self._render_3d_cuboids(frame, map_elements.get('cuboids', []), 
                               camera_params, ego_pose)
        
        return frame
    
    def _world_to_image(self, points_world: np.ndarray, camera_params: Dict,
                       ego_pose: np.ndarray) -> np.ndarray:
        ego_x, ego_y, ego_heading = ego_pose
        
        cos_h, sin_h = np.cos(ego_heading), np.sin(ego_heading)
        R = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
        
        t = np.array([ego_x, ego_y, 0])
        points_ego = (points_world - t) @ R.T
        
        cam_offset = camera_params.get('offset', np.array([0, 0, 1.5]))
        points_cam = points_ego - cam_offset
        
        fx = camera_params.get('fx', 1000)
        fy = camera_params.get('fy', 1000)
        cx = camera_params.get('cx', self.width / 2)
        cy = camera_params.get('cy', self.height / 2)
        
        u = fx * points_cam[:, 0] / (points_cam[:, 1] + 1e-6) + cx
        v = fy * points_cam[:, 2] / (points_cam[:, 1] + 1e-6) + cy
        
        return np.stack([u, v], axis=1)
    
    def _render_lane_lines(self, frame, lane_lines, camera_params, ego_pose):
        for lane in lane_lines:
            points_3d = np.array(lane['points'])
            lane_type = lane.get('type', 'white')
            
            if lane_type == 'yellow' or lane.get('no_pass', False):
                color = COSMOS_COLORS['lane_line_yellow']
            elif lane.get('dashed', False):
                color = COSMOS_COLORS['lane_line_dashed']
            else:
                color = COSMOS_COLORS['lane_line_white']
            
            points_2d = self._world_to_image(points_3d, camera_params, ego_pose)
            self._draw_polyline(frame, points_2d, color, COSMOS_LINE_WIDTHS['lane_line'])
    
    def _render_road_boundaries(self, frame, boundaries, camera_params, ego_pose):
        for boundary in boundaries:
            points_3d = np.array(boundary['points'])
            points_2d = self._world_to_image(points_3d, camera_params, ego_pose)
            self._draw_polyline(frame, points_2d, COSMOS_COLORS['road_boundary'],
                              COSMOS_LINE_WIDTHS['road_boundary'])
    
    def _render_crosswalks(self, frame, crosswalks, camera_params, ego_pose):
        for crosswalk in crosswalks:
            points_3d = np.array(crosswalk['points'])
            points_2d = self._world_to_image(points_3d, camera_params, ego_pose)
            self._draw_polyline(frame, points_2d, COSMOS_COLORS['crosswalk'],
                              COSMOS_LINE_WIDTHS['crosswalk'])
    
    def _render_road_markings(self, frame, markings, camera_params, ego_pose):
        for marking in markings:
            points_3d = np.array(marking['points'])
            points_2d = self._world_to_image(points_3d, camera_params, ego_pose)
            self._draw_polyline(frame, points_2d, COSMOS_COLORS['road_marking'],
                              COSMOS_LINE_WIDTHS['road_marking'])
    
    def _render_poles(self, frame, poles, camera_params, ego_pose):
        for pole in poles:
            pos_3d = np.array([pole['position']])
            pos_2d = self._world_to_image(pos_3d, camera_params, ego_pose)[0]
            
            if 0 <= pos_2d[0] < self.width and 0 <= pos_2d[1] < self.height:
                cv2.circle(frame, (int(pos_2d[0]), int(pos_2d[1])),
                          COSMOS_LINE_WIDTHS['pole'], COSMOS_COLORS['pole'], -1)
    
    def _render_traffic_elements(self, frame, traffic_lights, traffic_signs, 
                                 camera_params, ego_pose):
        for light in traffic_lights:
            pos_3d = np.array([light['position']])
            pos_2d = self._world_to_image(pos_3d, camera_params, ego_pose)[0]
            if 0 <= pos_2d[0] < self.width and 0 <= pos_2d[1] < self.height:
                cv2.circle(frame, (int(pos_2d[0]), int(pos_2d[1])),
                          COSMOS_LINE_WIDTHS['traffic_element'],
                          COSMOS_COLORS['traffic_light'], -1)
        
        for sign in traffic_signs:
            pos_3d = np.array([sign['position']])
            pos_2d = self._world_to_image(pos_3d, camera_params, ego_pose)[0]
            if 0 <= pos_2d[0] < self.width and 0 <= pos_2d[1] < self.height:
                cv2.circle(frame, (int(pos_2d[0]), int(pos_2d[1])),
                          COSMOS_LINE_WIDTHS['traffic_element'],
                          COSMOS_COLORS['traffic_sign'], -1)
    
    def _render_3d_cuboids(self, frame, cuboids, camera_params, ego_pose):
        for cuboid in cuboids:
            corners_3d = np.array(cuboid['corners'])
            corners_2d = self._world_to_image(corners_3d, camera_params, ego_pose)
            
            agent_type = cuboid.get('type', 'vehicle')
            color = COSMOS_COLORS.get(agent_type, COSMOS_COLORS['vehicle'])
            
            for i in range(4):
                self._draw_line(frame, corners_2d[i], corners_2d[(i+1)%4], 
                              color, COSMOS_LINE_WIDTHS['cuboid'])
            
            for i in range(4, 8):
                self._draw_line(frame, corners_2d[i], corners_2d[4+(i-4+1)%4], 
                              color, COSMOS_LINE_WIDTHS['cuboid'])
            
            for i in range(4):
                self._draw_line(frame, corners_2d[i], corners_2d[i+4], 
                              color, COSMOS_LINE_WIDTHS['cuboid'])
    
    def _draw_polyline(self, frame, points, color, thickness):
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            self._draw_line(frame, points[i], points[i+1], color, thickness)
    
    def _draw_line(self, frame, pt1, pt2, color, thickness):
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        if (0 <= x1 < self.width * 2 and 0 <= y1 < self.height * 2 and
            0 <= x2 < self.width * 2 and 0 <= y2 < self.height * 2):
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)


class EnrichedInterHubToCosmosConverter:
    """Enhanced converter from InterHub to Cosmos-Drive-Dreams format."""
    
    def __init__(self, interhub_cache_path: Union[str, Path],
                 dataset_name: str = "interaction_multi",
                 scene_dt: float = 0.1, verbose: bool = True,
                 image_width: int = 1280, image_height: int = 720):
        self.cache_path = Path(interhub_cache_path)
        self.dataset_name = dataset_name
        self.scene_dt = scene_dt
        self.verbose = verbose
        
        self.env_name = f"{dataset_name}"
        self.env_cache = EnvCache(self.cache_path)
        self.scenes_list = self.env_cache.load_env_scenes_list(self.env_name)
        
        self.renderer = CosmosMapRenderer(image_width, image_height)
        
        if self.verbose:
            print(f"✓ Loaded {len(self.scenes_list)} scenes from {self.dataset_name}")
    
    def extract_enriched_map_elements(self, scene: Scene, ego_pose: np.ndarray,
                                     clip_distance: float = 100.0) -> Dict:
        map_elements = {
            'lane_lines': [], 'road_boundaries': [], 'crosswalks': [],
            'road_markings': [], 'poles': [], 'traffic_lights': [], 'traffic_signs': [],
        }
        
        if not hasattr(scene, 'map_api') or scene.map_api is None:
            return map_elements
        
        vec_map = scene.map_api
        ego_x, ego_y = ego_pose[0], ego_pose[1]
        ego_pos = np.array([ego_x, ego_y])
        
        for lane in vec_map.lanes:
            points = np.array(lane.center.xy).T
            distances = np.linalg.norm(points - ego_pos, axis=1)
            if np.min(distances) > clip_distance:
                continue
            
            points_3d = np.column_stack([points, np.zeros(len(points))])
            map_elements['lane_lines'].append({
                'points': points_3d, 'type': 'white', 'dashed': False,
            })
        
        for road_edge in getattr(vec_map, 'road_edges', []):
            points = np.array(road_edge.xy).T
            distances = np.linalg.norm(points - ego_pos, axis=1)
            if np.min(distances) > clip_distance:
                continue
            
            points_3d = np.column_stack([points, np.zeros(len(points))])
            map_elements['road_boundaries'].append({'points': points_3d})
        
        return map_elements
    
    def extract_agent_cuboids(self, scene: Scene, scene_cache: DataFrameCache,
                             timestep: int, ego_pose: np.ndarray,
                             clip_distance: float = 80.0) -> List[Dict]:
        cuboids = []
        ego_pos = np.array([ego_pose[0], ego_pose[1]])
        
        column_dict = scene_cache.column_dict
        x_idx = column_dict['x']
        y_idx = column_dict['y']
        z_idx = column_dict.get('z', 2)
        heading_idx = column_dict['heading']
        length_idx = column_dict.get('length', None)
        width_idx = column_dict.get('width', None)
        
        if timestep < len(scene.agent_presence):
            agents_at_t = scene.agent_presence[timestep]
            
            for agent_info in agents_at_t:
                agent_name = agent_info.name
                
                try:
                    raw_state = scene_cache.get_raw_state(agent_id=agent_name, scene_ts=timestep)
                    if raw_state is None:
                        continue
                    
                    x = raw_state[x_idx]
                    y = raw_state[y_idx]
                    z = raw_state[z_idx] if z_idx < len(raw_state) else 0.0
                    heading = raw_state[heading_idx]
                    
                    agent_pos = np.array([x, y])
                    distance = np.linalg.norm(agent_pos - ego_pos)
                    if distance > clip_distance:
                        continue
                    
                    length = raw_state[length_idx] if length_idx and length_idx < len(raw_state) else 4.5
                    width = raw_state[width_idx] if width_idx and width_idx < len(raw_state) else 2.0
                    height = 1.5
                    
                    corners = self._generate_cuboid_corners(x, y, z, heading, length, width, height)
                    
                    agent_type = 'vehicle'
                    if hasattr(agent_info, 'agent_type'):
                        if agent_info.agent_type == AgentType.PEDESTRIAN:
                            agent_type = 'pedestrian'
                        elif agent_info.agent_type == AgentType.BICYCLE:
                            agent_type = 'bicycle'
                    
                    cuboids.append({
                        'corners': corners, 'type': agent_type, 'agent_id': agent_name,
                    })
                    
                except Exception as e:
                    if self.verbose:
                        print(f"[Cosmos] WARNING: Failed to extract cuboid for {agent_name}: {e}")
                    continue
        
        return cuboids
    
    def _generate_cuboid_corners(self, x: float, y: float, z: float, heading: float,
                                length: float, width: float, height: float) -> np.ndarray:
        l2, w2, h2 = length/2, width/2, height/2
        corners_local = np.array([
            [-l2, -w2, 0], [l2, -w2, 0], [l2, w2, 0], [-l2, w2, 0],
            [-l2, -w2, height], [l2, -w2, height], [l2, w2, height], [-l2, w2, height],
        ])
        
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        R = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
        
        corners_world = corners_local @ R.T + np.array([x, y, z])
        return corners_world
    
    def generate_hdmap_video(self, scene_idx: int, time_start: float, time_end: float,
                            output_dir: Union[str, Path], ego_agent_id: Optional[str] = None,
                            map_clip_distance: float = 100.0, agent_clip_distance: float = 80.0,
                            fps: int = 10) -> Path:
        """Generate Cosmos-Drive-Dreams compatible HDMap video."""
        if self.verbose:
            print(f"\n{'='*70}\nGenerating Cosmos-Drive-Dreams HDMap Video\n{'='*70}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        scene_tag = self.scenes_list[scene_idx]
        scene_name = getattr(scene_tag, "name", f"scene_{scene_idx}")
        scene: Scene = self.env_cache.load_scene(self.env_name, scene_name, scene_dt=self.scene_dt)
        
        scene_cache = DataFrameCache(cache_path=self.cache_path, scene=scene)
        column_dict = scene_cache.column_dict
        
        start_idx = int(time_start / scene.dt)
        end_idx = int(time_end / scene.dt)
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, scene.length_timesteps)
        
        if self.verbose:
            print(f"[Cosmos] Scene: {scene.name}")
            print(f"[Cosmos] Time window: [{time_start:.2f}, {time_end:.2f}] s")
            print(f"[Cosmos] Timesteps: {start_idx} to {end_idx}")
        
        # FIXED v3: Select ego agent with correct list handling
        if ego_agent_id is None:
            ego_agent_id, actual_start, actual_end = self._select_ego_agent(
                scene, scene_cache, column_dict, start_idx, end_idx
            )
            start_idx = actual_start
            end_idx = actual_end
        
        if self.verbose:
            print(f"[Cosmos] Ego agent: {ego_agent_id}")
            print(f"[Cosmos] Adjusted timesteps: {start_idx} to {end_idx}")
        
        camera_params = {
            'fx': 1000, 'fy': 1000,
            'cx': self.renderer.width / 2, 'cy': self.renderer.height / 2,
            'offset': np.array([1.5, 0, 1.5]),
        }
        
        video_path = output_path / f"hdmap_scene{scene_idx}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps,
                                (self.renderer.width, self.renderer.height))
        
        frames_generated = 0
        
        for t_idx in tqdm(range(start_idx, end_idx), desc="Generating HDMap frames",
                         disable=not self.verbose):
            try:
                raw_state = scene_cache.get_raw_state(agent_id=ego_agent_id, scene_ts=t_idx)
                
                if raw_state is None:
                    if self.verbose:
                        print(f"[Cosmos] WARNING: No data for ego at timestep {t_idx}")
                    continue
                
                ego_x = raw_state[column_dict['x']]
                ego_y = raw_state[column_dict['y']]
                ego_heading = raw_state[column_dict['heading']]
                    
            except Exception as e:
                if self.verbose:
                    print(f"[Cosmos] WARNING: Failed to get ego state at timestep {t_idx}: {e}")
                continue
            
            ego_pose = np.array([ego_x, ego_y, ego_heading])
            
            map_elements = self.extract_enriched_map_elements(scene, ego_pose, map_clip_distance)
            cuboids = self.extract_agent_cuboids(scene, scene_cache, t_idx, ego_pose, agent_clip_distance)
            map_elements['cuboids'] = cuboids
            
            frame = self.renderer.create_hdmap_frame(map_elements, camera_params, ego_pose)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            frames_generated += 1
        
        writer.release()
        
        if self.verbose:
            print(f"✓ HDMap video created: {video_path}")
            print(f"  Frames: {frames_generated}")
            print(f"  Duration: {frames_generated / fps:.2f} s")
            print(f"  Resolution: {self.renderer.width}x{self.renderer.height}")
            print(f"  File size: {video_path.stat().st_size / (1024*1024):.2f} MB")
        
        return video_path
    
    def _select_ego_agent(self, scene: Scene, scene_cache: DataFrameCache,
                         column_dict: Dict, start_idx: int, end_idx: int) -> Tuple[str, int, int]:
        """
        FIXED v3: Select ego agent handling scene.agents as a LIST.
        Returns: (agent_id, adjusted_start_idx, adjusted_end_idx)
        """
        if self.verbose:
            print(f"[Cosmos] Searching for suitable ego agent...")
        
        agent_durations = {}
        
        # FIXED: scene.agents is a LIST, not a dict
        for agent in scene.agents:
            agent_name = agent.name
            
            is_vehicle = (hasattr(agent, 'agent_type') and 
                         agent.agent_type == AgentType.VEHICLE)
            
            first_t = max(agent.first_timestep, start_idx)
            last_t = min(agent.last_timestep, end_idx - 1)
            
            if last_t >= first_t:
                duration = last_t - first_t + 1
                
                # Verify agent has data
                has_data = False
                try:
                    test_state = scene_cache.get_raw_state(agent_id=agent_name, scene_ts=first_t)
                    if test_state is not None:
                        has_data = True
                except:
                    pass
                
                if has_data:
                    agent_durations[agent_name] = {
                        'duration': duration, 'start': first_t, 'end': last_t + 1,
                        'is_vehicle': is_vehicle
                    }
        
        if not agent_durations:
            raise ValueError(
                f"No agents found in time range [{start_idx}, {end_idx}). "
                f"Scene has {len(scene.agents)} total agents."
            )
        
        # Sort by: 1) vehicle first, 2) longest duration
        sorted_agents = sorted(agent_durations.items(),
                             key=lambda x: (x[1]['is_vehicle'], x[1]['duration']),
                             reverse=True)
        
        ego_agent_id = sorted_agents[0][0]
        ego_info = agent_durations[ego_agent_id]
        
        if self.verbose:
            print(f"[Cosmos] Selected ego: {ego_agent_id}")
            print(f"[Cosmos]   Type: {'Vehicle' if ego_info['is_vehicle'] else 'Other'}")
            print(f"[Cosmos]   Duration: {ego_info['duration']} timesteps")
            print(f"[Cosmos]   Time range: [{ego_info['start']}, {ego_info['end']})")
        
        return ego_agent_id, ego_info['start'], ego_info['end']


def convert_interhub_to_cosmos_hdmap(cache_path: Union[str, Path], scene_idx: int,
                                     time_start: float, time_end: float,
                                     output_dir: Union[str, Path],
                                     dataset_name: str = "interaction_multi",
                                     ego_agent_id: Optional[str] = None,
                                     image_width: int = 1280, image_height: int = 720,
                                     fps: int = 10) -> Path:
    """Convenience function to generate Cosmos-Drive-Dreams HDMap video."""
    converter = EnrichedInterHubToCosmosConverter(
        interhub_cache_path=cache_path, dataset_name=dataset_name,
        verbose=True, image_width=image_width, image_height=image_height,
    )
    
    return converter.generate_hdmap_video(
        scene_idx=scene_idx, time_start=time_start, time_end=time_end,
        output_dir=output_dir, ego_agent_id=ego_agent_id, fps=fps,
    )