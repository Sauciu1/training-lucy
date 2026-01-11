from pdb import run
from pyexpat import model
from src.cluster_scripts import cluster_log_port
from src.cluster_scripts.cluster_log_port import DEFAULT_ENV_PARAMS, DEFAULT_MODEL_PARAMS, DEFAULT_RUN_PARAMS


if __name__ == "__main__":
    model_params = DEFAULT_MODEL_PARAMS

    run_params = DEFAULT_RUN_PARAMS
    run_params.update({"timesteps": 1_000_000})

    network_tests = [
        {"policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}},
        {"policy_kwargs": {"net_arch": {"pi": [512, 512], "vf": [512, 512]}}},
        {"policy_kwargs": {"net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]}}},
    ]


    for test_params in network_tests:
        model_params.update(test_params)
        cluster_log_port.main(model_params, output_prefix="walking_network_test")
