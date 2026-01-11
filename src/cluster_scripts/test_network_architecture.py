from pdb import run
from pyexpat import model
import sys
from src.cluster_scripts import cluster_log_port
from src.cluster_scripts.cluster_log_port import DEFAULT_ENV_PARAMS, DEFAULT_MODEL_PARAMS, DEFAULT_RUN_PARAMS


if __name__ == "__main__":
    model_params = DEFAULT_MODEL_PARAMS

    print("Starting network architecture tests...")

    run_params = DEFAULT_RUN_PARAMS
    run_params.update({"timesteps": 10_000_000, "env_number": 14})

    network_tests = [
        {"policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}},
        {"policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}},
        {"policy_kwargs": {"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]}}},
        {"policy_kwargs": {"net_arch": {"pi": [256, 256, 128, 64], "vf": [256, 256, 128, 64]}}},
        {"policy_kwargs": {"net_arch": {"pi": [512, 512], "vf": [512, 512]}}},
        {"policy_kwargs": {"net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]}}},
        {"policy_kwargs": {"net_arch": {"pi": [512, 512, 512, 128], "vf": [512, 512, 512, 256, 128]}}},

    ]


    for test_params in network_tests:
        model_params.update(test_params)
        cluster_log_port.main(model_params, run_params=run_params, output_prefix="long_n_arch_test")

    print("Network architecture tests completed.")
    sys.exit(0)
