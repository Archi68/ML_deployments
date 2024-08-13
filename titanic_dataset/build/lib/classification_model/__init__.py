import sys

sys.path.append(
    # "C:\\Users\\Irek9\\OneDrive\\Документы\\KV_DW_JL_5\\ML_deployments\\titanic_dataset\\"
    "C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments/titanic_dataset/"
)

import logging

from classification_model.config.core import PACKAGE_ROOT, config

logging.getLogger(config.app_config.package_name).addHandler(logging.NullHandler())

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()