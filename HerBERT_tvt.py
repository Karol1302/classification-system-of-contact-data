import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_scheduler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "model")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

LOG_FILE = os.path.join(OUTPUT_DIR, "training.log")
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "best_model.pth")
LOSS_PLOT_FILE = os.path.join(OUTPUT_DIR, "loss_plot.png")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_NAME = "allegro/herbert-base-cased"
ENABLE_NOISE = True

CLASS_COUNTS = {
    "imiona_meskie": 22230,
    "imiona_zenskie": 14680,
    "nazwiska": 29999,
    "miejscowosci": 29999,
    "ulice": 25864,
    "other": 29999
}
TOTAL_SAMPLES = sum(CLASS_COUNTS.values())
CLASS_WEIGHTS = {cls: TOTAL_SAMPLES / count for cls, count in CLASS_COUNTS.items()}

CATEGORY_WEIGHTS = {cls: TOTAL_SAMPLES / count for cls, count in CLASS_COUNTS.items()}

def preprocess_text(text, category):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    if category == "ulice":
        text = re.sub(r'^\s*(ul\.|al\.|pl\.)\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def add_noise_to_text(text, noise_level=0.2):
    if not isinstance(text, str):
        return text
    text = list(text)
    for _ in range(int(len(text) * noise_level)):
        idx = random.randint(0, len(text) - 1)
        if random.random() < 0.5:
            text[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
        else:
            del text[idx]
    return ''.join(text)

class TokenDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, weights=None, category=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.labels = labels
        self.weights = weights or [1.0] * len(texts)
        self.category = category
        if len(self.texts) != len(self.labels) or len(self.texts) != len(self.weights):
            raise ValueError(f"Nieprawidłowe dane: texts={len(self.texts)}, labels={len(self.labels)}, weights={len(self.weights)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx >= len(self.texts) or idx >= len(self.labels) or idx >= len(self.weights):
            raise IndexError(f"Indeks poza zakresem: idx={idx}, texts={len(self.texts)}, labels={len(self.labels)}, weights={len(self.weights)}")
        text = self.texts[idx]
        weight = self.weights[idx]
 
        if ENABLE_NOISE and self.category not in ["imiona_meskie", "imiona_zenskie"]:
            text = [add_noise_to_text(token) for token in text]
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokens['input_ids'][0]
        label_sequence = self.labels[idx][:self.max_length]
        label_ids = (label_sequence + [-100] * (self.max_length - len(label_sequence)))[:self.max_length]

        assert len(input_ids) == len(label_ids), "Mismatch between input_ids and labels length!"
        return {
            'input_ids': input_ids,
            'attention_mask': tokens['attention_mask'][0],
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'weight': torch.tensor(weight, dtype=torch.float)
        }

    
def plot_metrics(metrics):
    epochs = [m["epoch"] for m in metrics]
    train_losses = [m["train_loss"] for m in metrics]
    val_accuracies = [m["val_accuracy"] for m in metrics]
    val_f1_scores = [m["val_f1"] for m in metrics]

    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.title("Straty i metryki na przestrzeni epok")
    plt.xlabel("Epoki")
    plt.ylabel("Strata")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, val_accuracies, label="Val Accuracy", marker='o')
    plt.plot(epochs, val_f1_scores, label="Val F1 Score", marker='o')
    plt.xlabel("Epoki")
    plt.ylabel("Metryki")
    plt.legend()

    plt.tight_layout()
    metrics_plot_file = os.path.join(OUTPUT_DIR, "metrics_plot.png")
    plt.savefig(metrics_plot_file)
    logger.info(f"Wykres metryk zapisano do {metrics_plot_file}")

def normalize_weights(df, threshold=1000, category_weight=1.0):
    max_popularity = df[2].max()
    df[2] = df[2].apply(lambda x: min(x / threshold, 1.0) * category_weight)
    logger.info(f"Znormalizowano wagi w kategorii z maksymalną wartością: {max_popularity}")
    return df

def load_data(data_dir):
    logger.info("\n===== Wczytywanie danych =====")

    def load_from_folder(folder):
        texts, labels, weights = [], [], []
        categories = {
            "imiona_meskie": 0,
            "imiona_zenskie": 1,
            "nazwiska": 2,
            "miejscowosci": 3,
            "ulice": 4,
            "other": 5
        }
        folder_path = os.path.join(data_dir, folder)
        for category, label_id in categories.items():
            path = os.path.join(folder_path, f"{category}.csv")
            if not os.path.exists(path):
                logger.warning(f"Brak pliku: {path}")
                continue
            df = pd.read_csv(path, header=None)

            category_weight = CATEGORY_WEIGHTS[category]

            if df.shape[1] == 3:
                df = normalize_weights(df, threshold=1000, category_weight=category_weight)
                weights_to_add = df[2].tolist()
            elif df.shape[1] == 2:
                df[2] = 1 * category_weight
                weights_to_add = df[2].tolist()
            else:
                raise ValueError(f"Nieprawidłowa liczba kolumn w pliku {path}")

            df = df.dropna(subset=[0])
            df[0] = df[0].apply(str)
            df[0] = df[0].apply(lambda x: preprocess_text(x, category))

            texts_to_add = df[0].str.split().tolist()
            labels_to_add = [[label_id] * len(x.split()) for x in df[0] if isinstance(x, str)]

            if len(texts_to_add) != len(weights_to_add):
                weights_to_add = weights_to_add[:len(texts_to_add)]

            texts.extend(texts_to_add)
            labels.extend(labels_to_add)
            weights.extend(weights_to_add)

            logger.info(f"{category}: Wczytano {len(df)} wierszy z {folder}. Pierwszy wiersz: {df.iloc[0].tolist()}")

        if len(texts) != len(labels) or len(texts) != len(weights):
            raise ValueError(f"Nieprawidłowa długość danych: texts={len(texts)}, labels={len(labels)}, weights={len(weights)}")

        return texts, labels, weights

    train_texts, train_labels, train_weights = load_from_folder("training")
    val_texts, val_labels, val_weights = load_from_folder("validation")
    test_texts, test_labels, test_weights = load_from_folder("test")

    logger.info(f"Załadowano {len(train_texts)} przykładów treningowych.")
    logger.info(f"Załadowano {len(val_texts)} przykładów walidacyjnych.")
    logger.info(f"Załadowano {len(test_texts)} przykładów testowych.")
    return train_texts, train_labels, train_weights, val_texts, val_labels, val_weights, test_texts, test_labels, test_weights, {"imiona_meskie": 0, "imiona_zenskie": 1, "nazwiska": 2, "miejscowosci": 3, "ulice": 4, "other": 5}

def save_metrics(epoch, train_loss, val_acc, val_f1, val_precision, val_recall):
    with open(METRICS_FILE, 'a') as f:
        if epoch == 1:
            f.write("epoch,train_loss,val_accuracy,val_f1,val_precision,val_recall\n")
        f.write(f"{epoch},{train_loss:.4f},{val_acc:.4f},{val_f1:.4f},{val_precision:.4f},{val_recall:.4f}\n")

def evaluate_model(model, loader, categories, output_file):
    model.eval()
    predictions, true_labels, weights = [], [], []
    val_loss = 0

    for batch in loader:
        batch_weights = batch.pop('weight').to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            active_loss = batch['labels'].view(-1) != -100

            active_logits = outputs.logits.view(-1, model.num_labels)[active_loss]
            active_labels = batch['labels'].view(-1)[active_loss]

            expanded_weights = batch_weights.unsqueeze(1).expand(-1, batch['labels'].shape[1]).reshape(-1)
            active_weights = expanded_weights[active_loss]

            assert active_weights.shape[0] == active_logits.shape[0], "Mismatch in active_weights and logits shape!"

            loss = (loss_fct(active_logits, active_labels) * active_weights).mean()
            val_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        labels = batch['labels'].cpu().numpy()
        batch_weights = batch_weights.cpu().numpy()

        for i in range(labels.shape[0]):
            label_seq = labels[i]
            pred_seq = preds[i]
            weight_seq = batch_weights[i]
            for j, lbl in enumerate(label_seq):
                if lbl != -100:
                    true_labels.append(lbl)
                    predictions.append(pred_seq[j])
                    weights.append(weight_seq)

    logger.debug(f"true_labels length: {len(true_labels)}, predictions length: {len(predictions)}, weights length: {len(weights)}")

    accuracy = accuracy_score(true_labels, predictions, sample_weight=weights)
    f1 = f1_score(true_labels, predictions, average="weighted", sample_weight=weights)
    precision = precision_score(true_labels, predictions, average="weighted", sample_weight=weights)
    recall = recall_score(true_labels, predictions, average="weighted", sample_weight=weights)

    cm = confusion_matrix(true_labels, predictions, labels=list(categories.values()))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories.keys(), yticklabels=categories.keys())
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_file)
    logger.info(f"Macierz pomyłek zapisano do {output_file}")

    logger.info("Classification Report:\n" + classification_report(true_labels, predictions, target_names=list(categories.keys())))
    return val_loss / len(loader), accuracy, f1, precision, recall, cm


def train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, categories):
    metrics = []
    start_time = time.time()

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"===== Rozpoczynam Epokę {epoch}/{EPOCHS} =====")
        model.train()
        train_loss = 0
        for param in model.bert.encoder.layer[:6].parameters():
            param.requires_grad = epoch >= 3

        for step, batch in enumerate(train_loader):
            weights = batch.pop('weight').to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss * weights.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            if step % 50 == 0:
                logger.info(f"Step {step}/{len(train_loader)}: Current Loss={loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        epoch_confusion_matrix_file = os.path.join(OUTPUT_DIR, f"confusion_matrix_epoch_{epoch}.png")
        val_loss, val_acc, val_f1, val_precision, val_recall, _ = evaluate_model(model, val_loader, categories, epoch_confusion_matrix_file)

        logger.info(f"[Epoka {epoch}] Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, Val Accuracy={val_acc:.4f}, Val F1={val_f1:.4f}, Val Precision={val_precision:.4f}, Val Recall={val_recall:.4f}")

        if abs(avg_train_loss - val_acc) > 0.1:
            logger.warning(f"Możliwy overfitting! Train Loss={avg_train_loss:.4f}, Val Loss={val_acc:.4f}")

        scheduler.step(avg_train_loss)
        save_metrics(epoch, avg_train_loss, val_acc, val_f1, val_precision, val_recall)
        logger.info(f"[Epoka {epoch}] Train Loss={avg_train_loss:.4f}, Val Accuracy={val_acc:.4f}, Val F1={val_f1:.4f}, Val Precision={val_precision:.4f}, Val Recall={val_recall:.4f}")

        metrics.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_precision": val_precision,
            "val_recall": val_recall
        })

        if val_loss < best_val_loss:
            best_val_loss = avg_train_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
            logger.info("Zapisano nowy najlepszy model.")
        else:
            patience_counter += 1
            if patience_counter >= 4:
                logger.info("Early stopping triggered.")
                break
    plot_metrics(metrics)
    plot_losses(train_losses)

    end_time = time.time()
    train_time = end_time - start_time
    logger.info(f"Trening zakończono w czasie {train_time / 60:.2f} minut.")

def plot_losses(train_losses):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.title("Strata trenowania")
    plt.legend()
    plt.savefig(LOSS_PLOT_FILE)
    logger.info(f"Wykres strat zapisano do {LOSS_PLOT_FILE}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Pobieranie świeżego modelu {MODEL_NAME}.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR, num_labels=6)

    for idx, layer in enumerate(model.bert.encoder.layer):
        if idx < 6:
            layer.attention.self.dropout.p = 0.3
            layer.output.dropout.p = 0.3
        else:
            layer.attention.self.dropout.p = 0.2
            layer.output.dropout.p = 0.2

    train_texts, train_labels, train_weights, val_texts, val_labels, val_weights, test_texts, test_labels, test_weights, categories = load_data(DATA_DIR)

    train_dataset = TokenDataset(train_texts, train_labels, tokenizer, MAX_LENGTH, train_weights)
    val_dataset = TokenDataset(val_texts, val_labels, tokenizer, MAX_LENGTH, val_weights)
    test_dataset = TokenDataset(test_texts, test_labels, tokenizer, MAX_LENGTH, test_weights)

    if len(train_texts) != len(train_labels) or len(train_texts) != len(train_weights):
        raise ValueError("Nieprawidłowa długość danych treningowych.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.2)
    model.config.hidden_dropout_prob = 0.3

    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * EPOCHS)

    scaler = GradScaler()

    train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, categories)

    logger.info("===== Ewaluacja na zbiorze testowym =====")
    test_loss, test_acc, test_f1, test_precision, test_recall, _ = evaluate_model(model, test_loader, categories, os.path.join(OUTPUT_DIR, "test_confusion_matrix.png"))
    logger.info(f"[Test] Loss={test_loss:.4f}, Accuracy={test_acc:.4f}, F1={test_f1:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}")

    logger.info("Trening i testowanie zakończone!")
