from pathlib import Path

import mujoco
import mujoco.viewer


def main():
    scene_path = Path(__file__).resolve().with_name("scene.xml")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()