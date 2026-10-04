import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def main():
    parser = argparse.ArgumentParser(description="Train an Iris classifier.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    settings = parser.parse_args()

    flowers = load_iris()

    training_measurements, testing_measurements, training_answers, testing_answers = train_test_split(
        flowers.data,
        flowers.target,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=flowers.target
    )

    flower_model = DecisionTreeClassifier(
        random_state=settings.random_state
    )
    flower_model.fit(training_measurements, training_answers)

    predicted_answers = flower_model.predict(testing_measurements)
    accuracy = accuracy_score(testing_answers, predicted_answers)

    project_folder = Path(__file__).resolve().parent.parent
    output_folder = project_folder / "outputs"
    output_folder.mkdir(exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        testing_answers,
        predicted_answers,
        display_labels=flowers.target_names,
        cmap="Blues"
    )
    plt.title("Iris model: actual versus predicted flower types")
    plt.tight_layout()

    chart_file = output_folder / "confusion_matrix.png"
    plt.savefig(chart_file, dpi=150)
    plt.close()

    print("Training flowers:", len(training_measurements))
    print("Testing flowers:", len(testing_measurements))
    print(f"Accuracy: {accuracy:.1%}")
    print("Chart saved to:", chart_file)


if __name__ == "__main__":
    main()