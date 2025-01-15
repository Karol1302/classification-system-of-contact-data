import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans

data_dir = "data"
folders = ["training", "validation", "test"]
files = [
    "imiona_meskie.csv", "imiona_zenskie.csv",
    "nazwiska_meskie.csv", "nazwiska_zenskie.csv",
    "miejscowosci.csv", "ulice.csv"
]

results = []

def verify_proportions(file_name):
    counts = []
    for folder in folders:
        path = os.path.join(data_dir, folder, file_name)
        if os.path.exists(path):
            counts.append(len(pd.read_csv(path, header=0)))
        else:
            counts.append(0)

    total = sum(counts)
    if total > 0:
        for folder, count in zip(folders, counts):
            results.append({
                "File": file_name,
                "Category": "Proportions",
                "Metric": f"{folder.capitalize()} Proportion",
                "Value": f"{count / total:.2%} ({count} próbek)"
            })

def analyze_text_lengths(file_name):
    for folder in folders:
        path = os.path.join(data_dir, folder, file_name)
        if os.path.exists(path):
            lengths = pd.read_csv(path, header=0).iloc[:, 0].dropna().apply(len)
            results.append({
                "File": file_name,
                "Category": "Text Length",
                "Metric": f"{folder.capitalize()} Mean Length",
                "Value": f"{lengths.mean():.2f}"
            })
            results.append({
                "File": file_name,
                "Category": "Text Length",
                "Metric": f"{folder.capitalize()} Std Length",
                "Value": f"{lengths.std():.2f}"
            })

def check_uniqueness(file_name):
    data = {folder: set() for folder in folders}
    for folder in folders:
        path = os.path.join(data_dir, folder, file_name)
        if os.path.exists(path):
            data[folder] = set(pd.read_csv(path, header=0).iloc[:, 0].dropna())

    overlaps = {
        "Train-Validation Overlap": len(data["training"] & data["validation"]),
        "Train-Test Overlap": len(data["training"] & data["test"]),
        "Validation-Test Overlap": len(data["validation"] & data["test"]),
    }
    for metric, value in overlaps.items():
        results.append({
            "File": file_name,
            "Category": "Uniqueness",
            "Metric": metric,
            "Value": value
        })

def verify_clustering(file_name, n_clusters=5):
    path = os.path.join(data_dir, "training", file_name)
    if os.path.exists(path):
        lengths = pd.read_csv(path, header=0).iloc[:, 0].dropna().apply(len).values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(lengths)
        for i, count in Counter(kmeans.labels_).items():
            results.append({
                "File": file_name,
                "Category": "Clustering",
                "Metric": f"Cluster {i} Count",
                "Value": count
            })

def save_results_to_csv():
    output_file = "data_verification_results.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Wyniki zostały zapisane do: {output_file}")

def main():
    print("===== Weryfikacja jakości danych =====\n")
    for file_name in files:
        verify_proportions(file_name)
        analyze_text_lengths(file_name)
        check_uniqueness(file_name)
        verify_clustering(file_name)
    save_results_to_csv()

if __name__ == "__main__":
    main()
