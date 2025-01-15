import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import re
import pandas as pd
import csv
import re
import torch
from transformers import AutoTokenizer, BertForTokenClassification
import os
import threading

class DataCategorizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Categorization Tool")
        self.data = []
        self.processed_data = []
        self.log_messages = []
        self.selected_row_words = []
        self.selected_word = None

        self.setup_ui()

    def setup_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(top_frame, text="Import", command=self.import_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Process", command=self.process_data).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Export", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Undo", command=self.undo_action).pack(side=tk.LEFT, padx=5)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        input_frame = tk.Frame(main_frame)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(input_frame, text="Input Data:").pack(anchor=tk.W)
        self.tree = ttk.Treeview(input_frame, columns=("#1", "#2"), show="headings")
        self.tree.heading("#1", text="Select")
        self.tree.heading("#2", text="Input Data")

        self.tree.column("#1", width=50, anchor=tk.CENTER)
        self.tree.column("#2", width=300, anchor=tk.W)

        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree.bind("<Double-1>", self.open_word_selection_window)
        self.tree.bind("<Button-1>", self.toggle_checkbox)

        tk.Button(input_frame, text="Delete Selected", command=self.delete_selected_rows).pack(pady=5)

        entry_frame = tk.Frame(input_frame)
        entry_frame.pack(fill=tk.X, padx=5, pady=5)

        self.entry = tk.Entry(entry_frame)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(entry_frame, text="Add", command=self.add_manual_entry).pack(side=tk.LEFT, padx=5)

        processed_frame = tk.Frame(main_frame)
        processed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(processed_frame, text="Processed Data:").pack(anchor=tk.W)
        self.processed_tree = ttk.Treeview(processed_frame, columns=("#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10"), show="headings")
        self.processed_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for col, heading in enumerate(["Imię Męskie", "Imię Żeńskie",  "Nazwisko", "Numer PESEL", "Ulica", "Numer domu", "Kod pocztowy", "Miejscowość", "Numer telefonu", "Inne"], start=1):
            self.processed_tree.heading(f"#{col}", text=heading)

        log_frame = tk.Frame(self.root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(log_frame, text="Log:").pack(anchor=tk.W)
        self.log_text = tk.Text(log_frame, height=10, state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True)


    def delete_selected_rows(self):
        selected_items = [item for item in self.tree.get_children() if self.tree.set(item, "#1") == "1"]
        for item in selected_items:
            values = self.tree.item(item, "values")
            if values:
                self.data.remove(values[1])
            self.tree.delete(item)
        self.log(f"Deleted {len(selected_items)} selected rows.")

    def toggle_checkbox(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            current_value = self.tree.set(item, "#1")
            new_value = "1" if current_value == "0" else "0"
            self.tree.set(item, "#1", new_value)

    def log(self, message):
        self.log_messages.append(message)
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.configure(state="disabled")
        self.log_text.see(tk.END)

    def import_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            with open(file_path, "rb") as f:
                detected_encoding = "utf-8"
                try:

                    f.read().decode("utf-8")
                except UnicodeDecodeError:
                    detected_encoding = "cp1250"

            with open(file_path, newline="", encoding=detected_encoding) as csvfile:
                reader = csv.reader(csvfile, delimiter=";")
                self.data = [row[0].strip() for row in reader if row]

            self.update_table()
            self.log(f"Imported {len(self.data)} rows from {file_path} with encoding {detected_encoding}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import CSV: {e}")
            self.log(f"Error importing CSV: {e}")


    def update_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)

        for item in self.data:
            self.tree.insert("", "end", values=("0", item))

    def update_processed_table(self):
        for row in self.processed_tree.get_children():
            self.processed_tree.delete(row)

        for item in self.processed_data:
            self.processed_tree.insert("", "end", values=(
                item.get("Imię Męskie"),
                item.get("Imię Żeńskie"),
                item.get("Nazwisko"),
                item.get("Numer PESEL"),
                item.get("Ulica"),
                item.get("Numer domu"),
                item.get("Kod pocztowy"),
                item.get("Miejscowość"),
                item.get("Numer telefonu"),
                item.get("Inne")
            ))

    def process_data(self):
        self.processed_data = []
        for row in self.data:
            categorized_row = self.categorize_row(row)
            self.processed_data.append(categorized_row)
        self.update_processed_table()
        self.log("Data processing completed.")

    def word_to_tensor(self, tokens):
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded.")
        encoded = self.tokenizer(tokens, return_tensors="pt", truncation=True, padding=True, is_split_into_words=True)
        return encoded
    
    def categorize_row(self, row):

        original_row = row.strip()
        categorized = {
            "Imię Męskie": None,
            "Imię Żeńskie": None,
            "Nazwisko": None,
            "Ulica": None,
            "Numer domu": None,
            "Kod pocztowy": None,
            "Miejscowość": None,
            "Numer PESEL": None,
            "Numer telefonu": None,
            "Inne": None,
        }

        pesel = re.search(r"\b\d{11}\b", original_row)
        if pesel:
            categorized["Numer PESEL"] = pesel.group()
            original_row = original_row.replace(pesel.group(), "").strip()

        phone = re.search(r"\b(?:48\d{9}|\+48\s?\d{9}|\d{9})\b", original_row)
        if phone:
            phone_number = phone.group()
            categorized["Numer telefonu"] = phone_number
            original_row = original_row.replace(phone.group(), "").strip()

        postal_code = re.search(r"\b\d{2}-\d{3}\b", original_row)
        if postal_code:
            categorized["Kod pocztowy"] = postal_code.group()
            original_row = original_row.replace(postal_code.group(), "").strip()

        house_number = re.search(r"\b\d{1,5}[a-zA-Z]?(/\d{1,5})?\b", original_row)
        if house_number:
            categorized["Numer domu"] = house_number.group()
            original_row = original_row.replace(house_number.group(), "").strip()

        if self.model:
            tokens = original_row.split()
            if tokens:
                tensor_input = self.word_to_tensor(tokens)
                outputs = self.model(**tensor_input)
                predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

                label_map = {
                    0: "Imię Męskie",
                    1: "Imię Żeńskie",
                    2: "Nazwisko",
                    3: "Miejscowość",
                    4: "Ulica",
                    5: "Inne",
                }

                for token, label in zip(tokens, predictions):
                    category = label_map.get(label, "Inne")
                    if category == "Inne":
                        categorized["Inne"] = (categorized["Inne"] or "") + f" {token}"
                    else:
                        categorized[category] = token

        return categorized


    def export_csv(self):
        if not self.processed_data:
            messagebox.showinfo("Info", "No data to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            df = pd.DataFrame(self.processed_data)
            df.to_csv(file_path, sep=";", index=False)
            self.log(f"Exported data to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {e}")
            self.log(f"Error exporting CSV: {e}")

    def undo_action(self):
        if self.data:
            last_action = self.data.pop()
            self.update_table()
            self.log(f"Undo action: Removed last entry - {last_action}")

    def open_word_selection_window(self, event):
        item = self.tree.selection()
        if not item:
            return

        selected_row = self.tree.item(item[0], "values")[0]
        self.selected_row_words = selected_row.split()

        word_window = Toplevel(self.root)
        word_window.title("Select Word")

        tk.Label(word_window, text="Words in selected row:").pack(anchor=tk.W)
        word_list = tk.Listbox(word_window)
        word_list.pack(fill=tk.BOTH, expand=True)

        for word in self.selected_row_words:
            word_list.insert(tk.END, word)

        category_list = ["Imię", "Nazwisko", "Numer PESEL", "Ulica", "Numer domu", "Kod pocztowy", "Miejscowość", "Inne"]
        selected_category = tk.StringVar(value=category_list[0])

        tk.Label(word_window, text="Assign to category:").pack(anchor=tk.W)
        category_menu = ttk.Combobox(word_window, textvariable=selected_category, values=category_list, state="readonly")
        category_menu.pack(fill=tk.X, padx=5, pady=5)

        def assign_selected_word():
            selected = word_list.curselection()
            if selected:
                selected_word = word_list.get(selected)
                category = selected_category.get()
                self.log(f"Assigned word '{selected_word}' to category '{category}'")
                word_window.destroy()
                for item in self.processed_data:
                    if category in item and not item[category]:
                        item[category] = selected_word
                        break
                else:
                    self.processed_data.append({category: selected_word})
                self.update_processed_table()

        tk.Button(word_window, text="Assign", command=assign_selected_word).pack(pady=5)
    def delete_selected(self):
        selected_items = self.tree.selection()
        for item in selected_items:
            values = self.tree.item(item, "values")
            if values:
                self.data.remove(values[0])
            self.tree.delete(item)
        self.log(f"Deleted {len(selected_items)} selected rows.")

    def add_manual_entry(self):
        entry_text = self.entry.get().strip()
        if entry_text:
            self.data.append(entry_text)
            self.update_table()
            self.entry.delete(0, tk.END)
            self.log(f"Added manual entry: {entry_text}")

    def select_word(self, event):
        item = self.tree.selection()
        if not item:
            return

        selected_item = self.tree.item(item[0], "values")[0]
        self.selected_word = selected_item.split()
        self.log(f"Selected word: {self.selected_word}")

    def assign_to_category(self, event):
        if not hasattr(self, 'selected_word') or not self.selected_word:
            return

        region_id = self.processed_tree.identify_region(event.x, event.y)
        if region_id == "heading":
            col = self.processed_tree.identify_column(event.x)
            col_index = int(col.replace("#", "")) - 1

            category = ["Imię", "Nazwisko", "Numer PESEL", "Ulica", "Numer domu", "Kod pocztowy", "Miejscowość", "Inne"][col_index]

            word_to_assign = self.selected_word.pop(0)
            self.processed_data.append({category: word_to_assign})
            self.update_processed_table()
            self.log(f"Assigned '{word_to_assign}' to '{category}'")

    def __init__(self, root):
        self.root = root
        self.root.title("Data Categorization Tool")
        self.data = []
        self.processed_data = []
        self.log_messages = []
        self.model = None

        self.setup_ui()

        self.check_model()

    def check_model(self):
        model_path = "best_model.pth"
        if os.path.exists(model_path):
            threading.Thread(target=self.load_model, daemon=True).start()
        else:
            self.log(f"Model file {model_path} not found. Regex-based processing only.")
            self.model = None

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
            model_structure = BertForTokenClassification.from_pretrained(
                "allegro/herbert-base-cased",
                num_labels=7
            )

            state_dict = torch.load("best_model.pth")["model_state_dict"]

            corrected_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("classifier.1."):
                    corrected_key = key.replace("classifier.1.", "classifier.")
                else:
                    corrected_key = key
                corrected_state_dict[corrected_key] = value

            model_structure.load_state_dict(corrected_state_dict)
            model_structure.eval()
            self.model = model_structure
            self.log("Model loaded successfully.")
        except Exception as e:
            self.log(f"Failed to load model: {e}")
            self.model = None


if __name__ == "__main__":
    root = tk.Tk()
    app = DataCategorizationApp(root)
    root.mainloop()