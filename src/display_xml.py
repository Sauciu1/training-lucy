import mujoco
import mujoco.viewer
import os
import sys

from src import enforce_absolute_path
from .definitions import PROJECT_ROOT


def display_xml(xml_path: str, stance ='quad_stance'):


    if not os.path.isabs(xml_path):
        print("Making XML path absolute")
        print(f"  Before: {xml_path}")
        xml_path = os.path.join(PROJECT_ROOT, xml_path)
        print(f"After: {xml_path}")
        

    """Launch MuJoCo viewer to display the XML model."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    print(f"Loaded model: {xml_path}")
    print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")

    # Try to apply the named keyframe stance (if available), else fall back
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, stance)

    print(f"Applying keyframe '{stance}' (id={key_id}) for display")
    data.qpos[:] = model.key_qpos[key_id].copy()


    # Ensure forward kinematics are updated for correct visualization
    try:
        mujoco.mj_forward(model, data)
    except Exception:
        # mj_forward may not be available depending on Mujoco wrapper version
        pass

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python display_xml.py <path_to_xml>")


    xml_path = sys.argv[1]
    xml_path = enforce_absolute_path(xml_path)

    
    display_xml(xml_path)
