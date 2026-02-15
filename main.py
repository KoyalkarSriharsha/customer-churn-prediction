from src.feature_engineering import create_features
from src.train_model import train
from src.evaluate import evaluate_business_impact


def run_pipeline():

    print("Starting ML Pipeline...\n")

    create_features()
    train()
    evaluate_business_impact()

    print("\nPipeline Completed Successfully!")


if __name__ == "__main__":
    run_pipeline()
