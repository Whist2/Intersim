#!/usr/bin/env python3
"""
Simple Usage Example: Direct InterHub to RDS-HQ Conversion (FIXED VERSION)

This script demonstrates the simplest way to use the FIXED InterHub-TeraSim bridge
to generate RDS-HQ data directly from InterHub's unified cache.

Before running this script:
1. Install InterHub and run data unification to create the unified cache
2. Install TeraSim and ensure terasim-cosmos is available
3. Optionally run InterHub's interaction extraction for batch processing
4. Update the paths below to match your directory structure
"""

from interhub_terasim_bridge import InterHubToTeraSimBridge

# ==============================================================================
# Configuration - Update these paths for your setup
# ==============================================================================

# Path to InterHub's unified cache directory
# This is created by running: python 0_data_unify.py
INTERHUB_CACHE = "data/1_unified_cache"

# Dataset name in the cache
DATASET_NAME = "interaction_multi"

# Path to InterHub's interaction extraction results (optional, for batch mode)
# This is created by running: python 1_interaction_extract.py
INTERACTION_CSV = "data/2_extracted_results/results.csv"

# Where to save all RDS-HQ outputs
OUTPUT_DIR = "outputs/rds_hq_results"


# ==============================================================================
# Example 1: Convert a Single Scene
# ==============================================================================

def example_single_scene():
    """
    Convert a single scene from InterHub to RDS-HQ format.
    
    This is the simplest use case - you specify which scene to convert
    and the time window you're interested in. The bridge handles everything
    else automatically.
    """
    print("\n" + "="*70)
    print("Example 1: Single Scene Conversion")
    print("="*70 + "\n")
    
    # Initialize the bridge
    bridge = InterHubToTeraSimBridge(
        interhub_cache_path=INTERHUB_CACHE,
        dataset_name=DATASET_NAME,
        verbose=True  # Show detailed progress
    )
    
    # Generate RDS-HQ for a specific scene and time window
    output_path = bridge.generate_rds_hq_from_scene(
        scene_idx=10,              # Which scene from the dataset
        time_start=5.0,            # Start at 5 seconds into the scene
        time_end=15.0,             # End at 15 seconds into the scene
        output_dir=f"{OUTPUT_DIR}/example_1_single_scene",
        streetview_retrieval=False,  # Disable to speed up processing
        agent_clip_distance=80.0,    # Include agents within 80m of ego
    )
    
    print(f"\n✓ RDS-HQ data generated at: {output_path}")
    print("This directory can now be used with NVIDIA Cosmos-Drive!")


# ==============================================================================
# Example 2: Batch Process Multiple Interactions
# ==============================================================================

def example_batch_interactions():
    """
    Process multiple interaction scenarios identified by InterHub.
    
    This leverages InterHub's sophisticated interaction detection to focus
    on the most interesting segments of the data. Rather than converting
    entire hours of driving, we only process the interaction-rich segments.
    """
    print("\n" + "="*70)
    print("Example 2: Batch Processing from InterHub Interactions")
    print("="*70 + "\n")
    
    # Initialize the bridge
    bridge = InterHubToTeraSimBridge(
        interhub_cache_path=INTERHUB_CACHE,
        dataset_name=DATASET_NAME,
        verbose=True
    )
    
    # Process interactions from InterHub's extraction results
    output_dirs = bridge.batch_process_interactions(
        interaction_csv_path=INTERACTION_CSV,
        output_base_dir=f"{OUTPUT_DIR}/example_2_batch",
        max_scenes=5,  # Process first 5 interactions for testing
        streetview_retrieval=False,  # Disable for speed
        agent_clip_distance=80.0,    # Include agents within 80m
        camera_setting="default"     # Use default camera configuration
    )
    
    print(f"\n✓ Generated {len(output_dirs)} RDS-HQ datasets")
    print("\nOutput locations:")
    for i, output_dir in enumerate(output_dirs, 1):
        print(f"  {i}. {output_dir}")


# ==============================================================================
# Example 3: Explore Available Scenes
# ==============================================================================

def example_explore_scenes():
    """
    Explore what scenes are available before processing.
    
    This shows how to query scene information without actually converting
    anything, which is useful for understanding your dataset and deciding
    which scenes to process.
    """
    print("\n" + "="*70)
    print("Example 3: Exploring Available Scenes")
    print("="*70 + "\n")
    
    # Initialize the bridge
    bridge = InterHubToTeraSimBridge(
        interhub_cache_path=INTERHUB_CACHE,
        dataset_name=DATASET_NAME,
        verbose=False  # Disable verbose for cleaner output
    )
    
    # Show information about first 5 scenes
    print(f"Total scenes in dataset: {bridge.num_scenes}\n")
    print("First 5 scenes:")
    
    for scene_idx in range(min(5, bridge.num_scenes)):
        info = bridge.get_scene_info(scene_idx)
        
        print(f"\nScene {scene_idx}: {info['name']}")
        print(f"  Duration: {info['duration_seconds']:.2f} seconds")
        print(f"  Timesteps: {info['num_timesteps']}")
        print(f"  Number of agents: {info['num_agents']}")
        print(f"  Agent types: {info['agent_type_counts']}")


# ==============================================================================
# Example 4: Using the Convenience Function
# ==============================================================================

def example_convenience_function():
    """
    Use the simplified convenience function for one-off conversions.
    
    If you just need to convert a single scene and don't need to reuse
    the bridge object, you can use the convenience function which handles
    initialization automatically.
    """
    print("\n" + "="*70)
    print("Example 4: Using Convenience Function")
    print("="*70 + "\n")
    
    from interhub_terasim_bridge import convert_interhub_scene_to_rds_hq
    
    # Convert in one function call
    output_path = convert_interhub_scene_to_rds_hq(
        cache_path=INTERHUB_CACHE,
        scene_idx=15,
        time_start=0.0,
        time_end=10.0,
        output_dir=f"{OUTPUT_DIR}/example_4_convenience",
        dataset_name=DATASET_NAME,
        streetview_retrieval=False,
        agent_clip_distance=80.0,
    )
    
    print(f"\n✓ RDS-HQ data generated at: {output_path}")


# ==============================================================================
# Example 5: Custom Camera Settings
# ==============================================================================

def example_custom_camera():
    """
    Use custom camera configuration (e.g., Waymo-style cameras).
    """
    print("\n" + "="*70)
    print("Example 5: Custom Camera Configuration")
    print("="*70 + "\n")
    
    bridge = InterHubToTeraSimBridge(
        interhub_cache_path=INTERHUB_CACHE,
        dataset_name=DATASET_NAME,
        verbose=True
    )
    
    # Generate with Waymo camera configuration
    output_path = bridge.generate_rds_hq_from_scene(
        scene_idx=10,
        time_start=5.0,
        time_end=15.0,
        output_dir=f"{OUTPUT_DIR}/example_5_waymo_camera",
        camera_setting="waymo",  # Use Waymo camera setup
        streetview_retrieval=False,
        agent_clip_distance=80.0,
    )
    
    print(f"\n✓ RDS-HQ data with Waymo cameras generated at: {output_path}")


# ==============================================================================
# Example 6: Specific Agent Types
# ==============================================================================

def example_specific_ego():
    """
    Specify a particular agent as the ego vehicle.
    """
    print("\n" + "="*70)
    print("Example 6: Specifying Ego Vehicle")
    print("="*70 + "\n")
    
    bridge = InterHubToTeraSimBridge(
        interhub_cache_path=INTERHUB_CACHE,
        dataset_name=DATASET_NAME,
        verbose=True
    )
    
    # First, explore scene to find agent IDs
    scene_info = bridge.get_scene_info(10)
    print(f"Scene has {scene_info['num_agents']} agents")
    
    # Generate with specific ego (you'd get the actual agent ID from your data)
    output_path = bridge.generate_rds_hq_from_scene(
        scene_idx=10,
        time_start=5.0,
        time_end=15.0,
        output_dir=f"{OUTPUT_DIR}/example_6_specific_ego",
        ego_agent_id=None,  # Let it auto-select, or specify like "vehicle_001"
        streetview_retrieval=False,
        agent_clip_distance=80.0,
    )
    
    print(f"\n✓ RDS-HQ data generated at: {output_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run the examples."""
    import sys
    from pathlib import Path
    
    print("\n" + "="*70)
    print("InterHub to TeraSim Direct Integration Examples (FIXED VERSION)")
    print("="*70)
    
    # Check if required paths exist
    if not Path(INTERHUB_CACHE).exists():
        print(f"\n⚠ Error: InterHub cache not found at: {INTERHUB_CACHE}")
        print("Please run InterHub's data unification first:")
        print("  python 0_data_unify.py --desired_data interaction_multi \\")
        print("                         --load_path data/0_origin_datasets/interaction_multi \\")
        print("                         --save_path data/1_unified_cache")
        sys.exit(1)
    
    try:
        # Run example 1: Single scene conversion
        example_single_scene()
        
        # Run example 3: Explore scenes (doesn't require interaction CSV)
        example_explore_scenes()
        
        # Run example 4: Convenience function
        example_convenience_function()
        
        # Run example 5: Custom camera settings
        example_custom_camera()
        
        # Run example 6: Specific ego vehicle
        example_specific_ego()
        
        # Run example 2: Batch processing (only if interaction CSV exists)
        if Path(INTERACTION_CSV).exists():
            example_batch_interactions()
        else:
            print(f"\n⚠ Skipping batch example: {INTERACTION_CSV} not found")
            print("Run InterHub's interaction extraction to enable batch processing:")
            print("  python 1_interaction_extract.py --desired_data interaction_multi \\")
            print("                                  --cache_location data/1_unified_cache \\")
            print("                                  --save_path data/2_extracted_results")
        
        print("\n" + "="*70)
        print("✓ All Examples Complete!")
        print("="*70)
        print("\nThe generated RDS-HQ data can now be used with NVIDIA Cosmos-Drive")
        print("for training generative models and synthesizing sensor data.")
        print("\n✅ IMPORTANT: Check your output directories for:")
        print("   - cameras/ folder with 6 PNG files (6 camera views)")
        print("   - bbox_overlay/ with 3D bounding boxes")
        print("   - hdmap/ with HD map visualizations")
        
    except Exception as e:
        print(f"\n⚠ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()