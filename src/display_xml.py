import mujoco
import mujoco.viewer
import os
import sys

from src import enforce_absolute_path
from .definitions import PROJECT_ROOT


def display_xml(xml_path: str):


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
    

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python display_xml.py <path_to_xml>")


    xml_path = sys.argv[1]
    xml_path = enforce_absolute_path(xml_path)

    
    display_xml(xml_path)
