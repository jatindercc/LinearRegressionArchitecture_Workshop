# orchestrator_main.py

import argparse
import logging
from dotenv import load_dotenv
import os

from DataExtractionAnalysis.data_loader import DataExtractor
from DataPreparation.preprocessing import DataPreprocessor
from ModelSelection.model import ModelFactory
from ModelTraining.train import ModelTrainer
from ModelEvaluationValidation.evaluation import ModelEvaluator
from ModelTraining.stream_simulator import StreamSimulator


def main():
    load_dotenv()
    logging.basicConfig(filename="logs/app.log", level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str)
    parser.add_argument("--mode", choices=["stream", "batch", "realtime"], default="batch", help="Run mode: stream or batch")
    args = parser.parse_args()

    extractor = DataExtractor()
    preprocessor = DataPreprocessor()
    factory = ModelFactory()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()

    df = extractor.load_csv(args.train_csv)
    df = preprocessor.clean(df)
    X_train, X_test, y_train, y_test = preprocessor.split(df, target=os.getenv("TARGET_COLUMN"))

    model = factory.create_model()
    model = trainer.train(model, X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluator.evaluate(y_test, y_pred)
    print(metrics)

    evaluator.plot_residuals(y_test, y_pred)
    logging.info(f"Evaluation metrics: {metrics}")

    # Streaming simulation
    if args.mode == "stream":
        simulator = StreamSimulator(model, X_test, y_test, delay=1.0)
        simulator.run()


if __name__ == "__main__":
    main()