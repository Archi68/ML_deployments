import sys

sys.path.append("C:/Users/USER/OneDrive/Docs/KV_DW_JL_5/ML_deployments/die_casting")

from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.pipeline import die_casting_pipeline
from classification_model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """Загрузка данных, обучение модели и сохранение пайплайна"""
    data = load_dataset(file_name=config.app_config.raw_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    die_casting_pipeline.fit(X_train, y_train)

    save_pipeline(pipeline_to_persist=die_casting_pipeline)


if __name__ == "__main__":
    run_training()
