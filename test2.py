from pathlib import Path
import dill

# 你的 unified cache 根目录
CACHE_ROOT = Path("/mnt/e/intersim/interhub/data/1_unified_cache")
ENV_NAME = "interaction_multi"

scene_list_path = CACHE_ROOT / ENV_NAME / "scenes_list.dill"
print("scenes_list 路径:", scene_list_path)

with open(scene_list_path, "rb") as f:
    scenes = dill.load(f)

print(f"总场景数: {len(scenes)}")

first = scenes[0]
print("\n第一个场景的关键信息:")
for attr in dir(first):
    if attr.startswith("_"):
        continue
    try:
        value = getattr(first, attr)
    except Exception:
        continue
    if isinstance(value, (str, int, float, bool)):
        print(f"  {attr}: {value}")
