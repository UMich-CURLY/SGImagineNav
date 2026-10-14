#!/usr/bin/env python3
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch


def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the container with NVIDIA GPU access.")
    data = torch.ones((32, 32), device="cuda")
    torch.testing.assert_close(data @ data, torch.full_like(data, 32))
    print(f"CUDA: {torch.cuda.get_device_name(0)}")


def check_native_ops():
    from groundingdino import _C
    from pytorch3d.ops import box3d_overlap

    value = torch.ones((1, 4, 1, 8), device="cuda")
    shapes = torch.tensor([[2, 2]], dtype=torch.long, device="cuda")
    offsets = torch.tensor([0], dtype=torch.long, device="cuda")
    locations = torch.full((1, 1, 1, 1, 1, 2), 0.5, device="cuda")
    weights = torch.ones((1, 1, 1, 1, 1), device="cuda")
    output = _C.ms_deform_attn_forward(value, shapes, offsets, locations, weights, 1)
    torch.testing.assert_close(output, torch.ones_like(output))

    box = torch.tensor([[
        [0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1],
        [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1],
    ]], dtype=torch.float32, device="cuda")
    volume, overlap = box3d_overlap(box, box)
    torch.testing.assert_close(volume, torch.ones_like(volume))
    torch.testing.assert_close(overlap, torch.ones_like(overlap))
    print("GroundingDINO and PyTorch3D CUDA operators passed.")


def check_habitat():
    import habitat_sim
    from habitat.config.default import get_config

    task = get_config("configs/objectnav_hm3d_rgbd_semantic.yaml")
    assert task.habitat.environment.max_episode_steps == 500
    simulator = habitat_sim.SimulatorConfiguration()
    simulator.scene_id = "NONE"
    camera = habitat_sim.CameraSensorSpec()
    camera.uuid = "rgb"
    camera.sensor_type = habitat_sim.SensorType.COLOR
    camera.resolution = [16, 16]
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [camera]
    with habitat_sim.Simulator(habitat_sim.Configuration(simulator, [agent])) as sim:
        frame = sim.get_sensor_observations()["rgb"]
        assert frame.shape == (16, 16, 4)
    print("Habitat configuration and EGL rendering passed.")


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    for package in ["torch", "torchvision", "habitat-sim", "habitat-lab", "pytorch3d"]:
        print(f"{package}: {version(package)}")
    check_cuda()
    check_native_ops()
    check_habitat()
    with tempfile.TemporaryDirectory(prefix="sg-vln-check-") as config_dir:
        os.environ["YOLO_CONFIG_DIR"] = config_dir
        import main_vis_multi

        assert callable(main_vis_multi.main)
    print("SGImagineNav installation check passed.")


if __name__ == "__main__":
    main()
