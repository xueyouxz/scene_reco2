from pathlib import Path

from nuscenes_conventer.scene_mask_generator import SceneMaskGenerator


def main():
    """导出每个场景一个JSON文件的局部坐标系矢量地图。"""
    config_path = Path(__file__).resolve().parents[1] / "config" / "map_mask_config.yaml"
    generator = SceneMaskGenerator(str(config_path))
    version = generator.config["dataset_config"]["version"]
    splits_to_process = ["test"] if version == "test" else ["train", "val"]
    split_counts = generator.generate_scene_vector_maps(splits_to_process)

    for split, count in split_counts.items():
        print(f"{split}: 已导出 {count} 个场景矢量地图JSON")


if __name__ == "__main__":
    main()
