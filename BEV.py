"""
Bird's-Eye View Vehicle Selector with RDS-HQ Generation
(Cosmos-Drive-Dreams Style)

This application provides a GUI for:
1. Generating bird's-eye view from InterHub scenes
2. Manually selecting vehicles in the view
3. Generating RDS-HQ views for selected vehicles
4. Visualizing in Cosmos-Drive-Dreams Dark Mode standards
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QTextEdit, QSplitter, QComboBox, QProgressBar, QMessageBox,
    QFormLayout, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import traceback

# Import the bridge
# Ensure this path is correct relative to your setup
try:
    from interhub_terasim_bridge import InterHubToTeraSimBridge
except ImportError:
    # Fallback for direct execution if in same folder
    import sys
    sys.path.append('.')
    from interhub_terasim_bridge import InterHubToTeraSimBridge


class BirdsEyeViewCanvas(FigureCanvas):
    """
    Canvas for displaying bird's-eye view and handling vehicle selection.
    Uses Cosmos-Drive-Dreams color standards (Dark Mode).
    """
    
    # Cosmos Color Palette (Normalized 0-1 for Matplotlib)
    COLOR_BG = (0, 0, 0)
    COLOR_DRIVABLE = (40/255, 40/255, 40/255)
    COLOR_LANE = (255/255, 255/255, 255/255)
    COLOR_HIGHLIGHT = (0, 1, 1)  # Cyan for selection
    COLOR_TEXT = (1, 1, 1)       # White text
    
    def __init__(self, parent=None):
        # Set dark background for figure
        self.fig = Figure(figsize=(10, 10), facecolor=self.COLOR_BG)
        self.ax = self.fig.add_subplot(111)
        # Set dark background for axis
        self.ax.set_facecolor(self.COLOR_BG)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Data storage
        self.vehicle_data = {}  # {vehicle_id: {'x': [], 'y': [], 'times': []}}
        self.selected_vehicle = None
        self.vehicle_artists = {}  # {vehicle_id: artist}
        self.trajectory_artists = {}
        self.highlight_artist = None
        
        # Connect mouse click event
        self.mpl_connect('button_press_event', self.on_click)
        
        self.setup_plot()
    
    def setup_plot(self):
        """Initialize the plot with Cosmos styling."""
        self.ax.clear()
        
        # Styling for Cosmos Dark Mode
        self.ax.set_facecolor(self.COLOR_BG)
        self.ax.set_title("Bird's-Eye View (Cosmos Standard)", 
                         fontsize=14, fontweight='bold', color=self.COLOR_TEXT)
        self.ax.set_xlabel("X Position (m)", fontsize=12, color=self.COLOR_TEXT)
        self.ax.set_ylabel("Y Position (m)", fontsize=12, color=self.COLOR_TEXT)
        
        # Adjust tick colors
        self.ax.tick_params(axis='x', colors=self.COLOR_TEXT)
        self.ax.tick_params(axis='y', colors=self.COLOR_TEXT)
        
        # Update spines (borders)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.COLOR_TEXT)
            
        self.ax.set_aspect('equal')
        # Faint grid mimicking drivable area texture
        self.ax.grid(True, alpha=0.2, color='white', linestyle='--')
        self.draw()
    
    def load_trajectory_data(self, fcd_path: Path, time_start: float, time_end: float):
        """
        Load vehicle trajectory data from FCD file.
        """
        self.vehicle_data.clear()
        
        try:
            tree = ET.parse(fcd_path)
            root = tree.getroot()
            
            for timestep in root.findall('timestep'):
                time = float(timestep.get('time'))
                
                # Filter by time range
                if time < time_start or time > time_end:
                    continue
                
                for vehicle in timestep.findall('vehicle'):
                    veh_id = vehicle.get('id')
                    x = float(vehicle.get('x'))
                    y = float(vehicle.get('y'))
                    angle = float(vehicle.get('angle', 0))
                    speed = float(vehicle.get('speed', 0))
                    
                    if veh_id not in self.vehicle_data:
                        self.vehicle_data[veh_id] = {
                            'x': [], 'y': [], 'times': [], 
                            'angles': [], 'speeds': []
                        }
                    
                    self.vehicle_data[veh_id]['x'].append(x)
                    self.vehicle_data[veh_id]['y'].append(y)
                    self.vehicle_data[veh_id]['times'].append(time)
                    self.vehicle_data[veh_id]['angles'].append(angle)
                    self.vehicle_data[veh_id]['speeds'].append(speed)
                    
            return len(self.vehicle_data)
        except Exception as e:
            print(f"Error loading FCD: {e}")
            return 0
    
    def plot_trajectories(self, current_time: Optional[float] = None):
        """
        Plot all vehicle trajectories and current positions.
        """
        self.ax.clear()
        # Re-apply Cosmos styling after clear
        self.ax.set_facecolor(self.COLOR_BG)
        self.ax.tick_params(colors=self.COLOR_TEXT)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.COLOR_TEXT)
        
        self.vehicle_artists.clear()
        self.trajectory_artists.clear()
        
        if not self.vehicle_data:
            self.ax.text(0.5, 0.5, "No trajectory data loaded",
                        ha='center', va='center', transform=self.ax.transAxes,
                        fontsize=14, color=self.COLOR_TEXT)
            self.draw()
            return
        
        # Plot each vehicle's trajectory
        for veh_id, data in self.vehicle_data.items():
            xs = np.array(data['x'])
            ys = np.array(data['y'])
            times = np.array(data['times'])
            
            if len(xs) == 0:
                continue
            
            # Plot trajectory line - Using White/Grey to mimic lane markings in Cosmos
            line, = self.ax.plot(xs, ys, '-', alpha=0.4, linewidth=1, 
                                color=self.COLOR_LANE,
                                label=veh_id if len(self.vehicle_data) <= 5 else None)
            self.trajectory_artists[veh_id] = line
            
            # Plot current position if time is specified
            if current_time is not None:
                # Find closest time index
                time_diffs = np.abs(times - current_time)
                if len(time_diffs) > 0:
                    idx = np.argmin(time_diffs)
                    
                    # Plot vehicle as a marker
                    marker = self.ax.scatter(xs[idx], ys[idx], s=60, 
                                           c='white',  # Vehicles are white
                                           marker='o', 
                                           edgecolors='gray', 
                                           alpha=0.8,
                                           linewidths=1.0,
                                           zorder=5,
                                           picker=5)  # Enable picking
                    self.vehicle_artists[veh_id] = marker
            else:
                # If no time specified, show all positions roughly
                marker = self.ax.scatter(xs, ys, s=30, 
                                       c='white',
                                       alpha=0.3,
                                       marker='o',
                                       zorder=5,
                                       picker=5)
                self.vehicle_artists[veh_id] = marker
        
        # Highlight selected vehicle
        if self.selected_vehicle and self.selected_vehicle in self.vehicle_data:
            self.highlight_selected_vehicle()
        
        self.ax.set_title(f"Bird's-Eye View ({len(self.vehicle_data)} vehicles)", 
                         fontsize=14, fontweight='bold', color=self.COLOR_TEXT)
        self.ax.set_xlabel("X Position (m)", fontsize=12, color=self.COLOR_TEXT)
        self.ax.set_ylabel("Y Position (m)", fontsize=12, color=self.COLOR_TEXT)
        self.ax.grid(True, alpha=0.15, color='white', linestyle='--')
        
        self.draw()
    
    def highlight_selected_vehicle(self):
        """Highlight the currently selected vehicle."""
        if not self.selected_vehicle or self.selected_vehicle not in self.vehicle_data:
            return
        
        data = self.vehicle_data[self.selected_vehicle]
        xs = np.array(data['x'])
        ys = np.array(data['y'])
        
        # Remove old highlight
        if self.highlight_artist:
            self.highlight_artist.remove()
        
        # Create highlight circle around the trajectory center
        center_x = np.mean(xs)
        center_y = np.mean(ys)
        radius = max(np.std(xs), np.std(ys)) * 2 + 5
        
        # Use Cyan for highlight (Cosmos high contrast)
        highlight_circle = Circle((center_x, center_y), radius, 
                                 fill=False, edgecolor=self.COLOR_HIGHLIGHT, 
                                 linewidth=2, linestyle='--',
                                 zorder=10)
        self.highlight_artist = self.ax.add_patch(highlight_circle)
        
        # Make trajectory thicker and brighter
        if self.selected_vehicle in self.trajectory_artists:
            line = self.trajectory_artists[self.selected_vehicle]
            line.set_linewidth(3)
            line.set_color(self.COLOR_HIGHLIGHT)
            line.set_alpha(1.0)
            line.set_zorder(15)
        
        self.draw()
    
    def on_click(self, event):
        """Handle mouse click to select vehicle."""
        if event.inaxes != self.ax:
            return
        
        if not self.vehicle_data:
            return
        
        # Find the closest vehicle to click position
        click_x, click_y = event.xdata, event.ydata
        min_dist = float('inf')
        closest_vehicle = None
        
        for veh_id, data in self.vehicle_data.items():
            xs = np.array(data['x'])
            ys = np.array(data['y'])
            
            if len(xs) == 0:
                continue
            
            distances = np.sqrt((xs - click_x)**2 + (ys - click_y)**2)
            min_veh_dist = np.min(distances)
            
            if min_veh_dist < min_dist:
                min_dist = min_veh_dist
                closest_vehicle = veh_id
        
        # Select vehicle if click is close enough (within 10 meters)
        if closest_vehicle and min_dist < 10:
            self.select_vehicle(closest_vehicle)
    
    def select_vehicle(self, vehicle_id: str):
        """
        Select a specific vehicle.
        Args:
            vehicle_id: ID of the vehicle to select
        """
        if vehicle_id not in self.vehicle_data:
            return
        
        # Deselect previous vehicle visual
        if self.selected_vehicle and self.selected_vehicle in self.trajectory_artists:
            line = self.trajectory_artists[self.selected_vehicle]
            line.set_linewidth(1)
            line.set_color(self.COLOR_LANE) # Revert to white/lane color
            line.set_alpha(0.4)
        
        self.selected_vehicle = vehicle_id
        self.highlight_selected_vehicle()
        
        print(f"Selected vehicle: {vehicle_id}")


class RDSHQGenerationThread(QThread):
    """Thread for generating RDS-HQ data in the background."""
    
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)  # Path to RDS-HQ directory
    error = pyqtSignal(str)
    
    def __init__(self, bridge, scene_idx, time_start, time_end, 
                 output_dir, ego_agent_id, streetview_retrieval, 
                 agent_clip_distance, camera_setting):
        super().__init__()
        self.bridge = bridge
        self.scene_idx = scene_idx
        self.time_start = time_start
        self.time_end = time_end
        self.output_dir = output_dir
        self.ego_agent_id = ego_agent_id
        self.streetview_retrieval = streetview_retrieval
        self.agent_clip_distance = agent_clip_distance
        self.camera_setting = camera_setting
    
    def run(self):
        try:
            self.progress.emit("Generating RDS-HQ data...")
            
            # The bridge handles the Cosmos color config automatically via its init defaults
            rds_path = self.bridge.generate_rds_hq_from_scene(
                scene_idx=self.scene_idx,
                time_start=self.time_start,
                time_end=self.time_end,
                output_dir=self.output_dir,
                ego_agent_id=self.ego_agent_id,
                streetview_retrieval=self.streetview_retrieval,
                agent_clip_distance=self.agent_clip_distance,
                camera_setting=self.camera_setting
            )
            
            self.progress.emit("RDS-HQ generation complete!")
            self.finished.emit(str(rds_path))
            
        except Exception as e:
            error_msg = f"Error generating RDS-HQ: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.bridge = None
        self.current_fcd_path = None
        self.current_scene_info = None
        
        self.init_ui()
        self.setWindowTitle("Bird's-Eye View & RDS-HQ Generator (Cosmos Enhanced)")
        self.setGeometry(100, 100, 1400, 900)
    
    def init_ui(self):
        """Initialize the user interface."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        left_panel = self.create_control_panel()
        right_panel = self.create_visualization_panel()
        
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
    
    def create_control_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # InterHub Configuration
        config_group = QGroupBox("InterHub Configuration")
        config_layout = QFormLayout()
        
        self.cache_path_label = QLabel("Not loaded")
        config_layout.addRow("Cache Path:", self.cache_path_label)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["interaction_multi", "waymo", "nuplan"])
        config_layout.addRow("Dataset:", self.dataset_combo)
        
        self.load_bridge_btn = QPushButton("Load InterHub Data")
        self.load_bridge_btn.clicked.connect(self.load_interhub_bridge)
        config_layout.addRow(self.load_bridge_btn)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Scene Selection
        scene_group = QGroupBox("Scene Selection")
        scene_layout = QFormLayout()
        
        self.scene_idx_spin = QSpinBox()
        self.scene_idx_spin.setRange(0, 10000)
        self.scene_idx_spin.setValue(0)
        scene_layout.addRow("Scene Index:", self.scene_idx_spin)
        
        self.time_start_spin = QDoubleSpinBox()
        self.time_start_spin.setRange(0, 1000)
        self.time_start_spin.setValue(0.0)
        self.time_start_spin.setSingleStep(1.0)
        scene_layout.addRow("Time Start (s):", self.time_start_spin)
        
        self.time_end_spin = QDoubleSpinBox()
        self.time_end_spin.setRange(0, 1000)
        self.time_end_spin.setValue(10.0)
        self.time_end_spin.setSingleStep(1.0)
        scene_layout.addRow("Time End (s):", self.time_end_spin)
        
        self.load_scene_btn = QPushButton("Load Scene & Generate View")
        self.load_scene_btn.clicked.connect(self.load_scene)
        self.load_scene_btn.setEnabled(False)
        scene_layout.addRow(self.load_scene_btn)
        
        scene_group.setLayout(scene_layout)
        layout.addWidget(scene_group)
        
        # Selected Vehicle Info
        vehicle_group = QGroupBox("Selected Vehicle")
        vehicle_layout = QVBoxLayout()
        
        self.selected_vehicle_label = QLabel("None")
        self.selected_vehicle_label.setStyleSheet("font-weight: bold; color: blue;")
        vehicle_layout.addWidget(self.selected_vehicle_label)
        
        self.vehicle_info_text = QTextEdit()
        self.vehicle_info_text.setReadOnly(True)
        self.vehicle_info_text.setMaximumHeight(100)
        vehicle_layout.addWidget(self.vehicle_info_text)
        
        vehicle_group.setLayout(vehicle_layout)
        layout.addWidget(vehicle_group)
        
        # RDS-HQ Generation
        rds_group = QGroupBox("RDS-HQ Generation")
        rds_layout = QFormLayout()
        
        self.streetview_check = QCheckBox()
        self.streetview_check.setChecked(False)
        rds_layout.addRow("Street View:", self.streetview_check)
        
        self.agent_clip_spin = QDoubleSpinBox()
        self.agent_clip_spin.setRange(10, 200)
        self.agent_clip_spin.setValue(80.0)
        self.agent_clip_spin.setSingleStep(10)
        rds_layout.addRow("Agent Clip (m):", self.agent_clip_spin)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["default", "waymo"])
        rds_layout.addRow("Camera Setting:", self.camera_combo)
        
        self.generate_rds_btn = QPushButton("Generate RDS-HQ Video")
        self.generate_rds_btn.clicked.connect(self.generate_rds_hq)
        self.generate_rds_btn.setEnabled(False)
        # Style the button to indicate action
        self.generate_rds_btn.setStyleSheet("background-color: #d1e7dd; font-weight: bold;")
        rds_layout.addRow(self.generate_rds_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        rds_layout.addRow(self.progress_bar)
        
        rds_group.setLayout(rds_layout)
        layout.addWidget(rds_group)
        
        # Status Log
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self) -> QWidget:
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title = QLabel("Bird's-Eye View Visualization")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Canvas
        self.canvas = BirdsEyeViewCanvas()
        layout.addWidget(self.canvas)
        
        # Instructions
        instructions = QLabel(
            "Instructions:\n"
            "1. Load InterHub data.\n"
            "2. Load Scene (Creates temporary FCD).\n"
            "3. Click vehicle on dark map to select (Cyan Highlight).\n"
            "4. Generate RDS-HQ (Uses Cosmos Colors)."
        )
        instructions.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(instructions)
        
        return panel
    
    def log(self, message: str):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def load_interhub_bridge(self):
        try:
            self.log("Loading InterHub bridge...")
            # Modify this path to where your unified cache actually is
            cache_path = "data/1_unified_cache" 
            
            # Check if directory exists, otherwise warn (mock for GUI testing if needed)
            if not Path(cache_path).exists():
                 self.log(f"WARNING: Cache path {cache_path} not found. Ensure 0_data_unify.py ran.")
            
            self.bridge = InterHubToTeraSimBridge(
                interhub_cache_path=cache_path,
                dataset_name=self.dataset_combo.currentText(),
                verbose=True
            )
            
            self.cache_path_label.setText(cache_path)
            self.scene_idx_spin.setRange(0, self.bridge.num_scenes - 1)
            
            self.log(f"✓ Successfully loaded {self.bridge.num_scenes} scenes")
            self.load_scene_btn.setEnabled(True)
            
        except Exception as e:
            self.log(f"✗ Error loading InterHub: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load InterHub:\n{str(e)}")
    
    def load_scene(self):
        if not self.bridge:
            QMessageBox.warning(self, "Warning", "Please load InterHub data first!")
            return
        
        try:
            scene_idx = self.scene_idx_spin.value()
            time_start = self.time_start_spin.value()
            time_end = self.time_end_spin.value()
            
            self.log(f"\nLoading scene {scene_idx} [{time_start:.1f}s - {time_end:.1f}s]...")
            
            # Create a temp directory for this visualization session
            output_dir = Path("temp_vis_data") / f"scene_{scene_idx}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.log("Generating trajectory data (FCD)...")
            
            # We call the internal method to just get FCD for visualization
            # without running the full RDS pipeline yet
            # Accessing the internal method _generate_sumo_fcd directly is efficient here
            # But safer to use the public API if possible. 
            # We'll use the bridge's FCD generation logic.
            
            scene = self.bridge.env_cache.load_scene(
                self.bridge.env_name, 
                self.bridge.scenes_list[scene_idx].name, 
                scene_dt=self.bridge.scene_dt
            )
            
            fcd_path, _, _ = self.bridge._generate_sumo_fcd(
                scene=scene,
                time_start=time_start,
                time_end=time_end,
                output_dir=output_dir,
                ego_agent_id=None, # We want all vehicles to start
                agent_clip_distance=self.agent_clip_spin.value()
            )
            
            self.current_fcd_path = fcd_path
            
            # Load into Canvas
            self.log("Rendering on Cosmos-Style Canvas...")
            num_vehicles = self.canvas.load_trajectory_data(
                self.current_fcd_path, time_start, time_end
            )
            
            self.canvas.plot_trajectories(current_time=(time_start + time_end) / 2)
            
            self.current_scene_info = {
                'scene_idx': scene_idx,
                'time_start': time_start,
                'time_end': time_end,
                'output_dir': output_dir
            }
            
            self.log(f"✓ Displaying {num_vehicles} vehicles.")
            
        except Exception as e:
            self.log(f"✗ Error loading scene: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load scene:\n{str(e)}")
    
    def update_selected_vehicle_info(self):
        if not self.canvas.selected_vehicle:
            self.selected_vehicle_label.setText("None")
            self.vehicle_info_text.clear()
            self.generate_rds_btn.setEnabled(False)
            return
        
        veh_id = self.canvas.selected_vehicle
        data = self.canvas.vehicle_data[veh_id]
        
        self.selected_vehicle_label.setText(veh_id)
        
        xs = np.array(data['x'])
        ys = np.array(data['y'])
        
        info = f"ID: {veh_id}\nFrames: {len(xs)}\nX range: {xs.min():.1f} to {xs.max():.1f}"
        self.vehicle_info_text.setText(info)
        self.generate_rds_btn.setEnabled(True)
    
    def generate_rds_hq(self):
        """Generate final RDS-HQ output."""
        if not self.canvas.selected_vehicle:
            return
        
        try:
            self.log(f"\nGenerating RDS-HQ for {self.canvas.selected_vehicle}...")
            self.generate_rds_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            
            # Output folder
            out_folder = f"rds_output_{self.canvas.selected_vehicle}"
            
            self.rds_thread = RDSHQGenerationThread(
                bridge=self.bridge,
                scene_idx=self.current_scene_info['scene_idx'],
                time_start=self.current_scene_info['time_start'],
                time_end=self.current_scene_info['time_end'],
                output_dir=out_folder,
                ego_agent_id=self.canvas.selected_vehicle,
                streetview_retrieval=self.streetview_check.isChecked(),
                agent_clip_distance=self.agent_clip_spin.value(),
                camera_setting=self.camera_combo.currentText()
            )
            
            self.rds_thread.progress.connect(self.log)
            self.rds_thread.finished.connect(self.on_rds_generation_finished)
            self.rds_thread.error.connect(self.on_rds_generation_error)
            
            self.rds_thread.start()
            
        except Exception as e:
            self.log(f"✗ Error: {e}")
            self.generate_rds_btn.setEnabled(True)

    def on_rds_generation_finished(self, rds_path: str):
        self.progress_bar.setVisible(False)
        self.generate_rds_btn.setEnabled(True)
        self.log(f"✓ RDS-HQ Generated in: {rds_path}")
        QMessageBox.information(self, "Success", f"RDS-HQ generated!\nPath: {rds_path}")

    def on_rds_generation_error(self, error_msg: str):
        self.progress_bar.setVisible(False)
        self.generate_rds_btn.setEnabled(True)
        self.log(f"✗ Failed: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    
    # Hook up selection event
    original_select = window.canvas.select_vehicle
    def select_with_update(vehicle_id):
        original_select(vehicle_id)
        window.update_selected_vehicle_info()
    window.canvas.select_vehicle = select_with_update
    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()