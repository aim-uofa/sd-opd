"""Build the environment forwarded to Ray training workers."""

import json
import os


def main() -> None:
    keys = (
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "SD_OPD_RAY_RUN_ID",
        "TENSORBOARD_DIR",
    )
    print(json.dumps({"env_vars": {key: os.environ[key] for key in keys}}))


if __name__ == "__main__":
    main()
