
import os
import sys
import json
from typing import Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import render_model_gym

def extract_paths_from_runinfo(run_info_path: str) -> Tuple[str, str, str]:
	"""Extract model_path, xml_path, and env_type from run_info.json."""
	with open(run_info_path, 'r', encoding='utf-8') as f:
		info = json.load(f)
	model_path = info['artifacts']['model_path']
	xml_path = info['env_params']['env_kwargs']['xml_file']
	env_type = 'lucy'  # Could be made dynamic if needed
	return model_path, xml_path, env_type

def run_gym_from_runinfo(run_info_path: str, speed: float = 1.0):
	model_path, xml_path, env_type = extract_paths_from_runinfo(run_info_path)
	print(f"Loading model: {model_path}\nLoading XML: {xml_path}\nEnv type: {env_type}\nSpeed: {speed}")
	render_model_gym.render_model_gym(model_path, xml_path, env_type, speed)

if __name__ == "__main__":
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		print("Usage: python cluster_gym.py <path_to_run_info.json> [speed]")
		sys.exit(1)
	run_info_path = sys.argv[1]
	speed = float(sys.argv[2]) if len(sys.argv) == 3 else 1.0
	run_gym_from_runinfo(run_info_path, speed)
