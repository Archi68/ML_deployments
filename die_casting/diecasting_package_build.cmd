cd C:\Users\USER\OneDrive\Docs\KV_DW_JL_5\ML_deployments\die_casting
py -m pip install --upgrade build
py -m build
py -m pip install .
cd C:\Users\USER\OneDrive\Docs\KV_DW_JL_5\ML_deployments\die_casting\dist
pip install die_casting_classification_model-0.0.2-py3-none-any.whl