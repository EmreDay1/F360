import os
import json
import re
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    pipeline
)


os.environ["WANDB_DISABLED"] = "true"


def format_to_jsonable(input_string):
    try:
        structured_data = {}
        parameters = {}

     
        shape_match = re.search(r'"shape":\s*"([^"]+)"', input_string)
        if shape_match:
            structured_data['shape'] = shape_match.group(1).strip()

        parameters_match = re.search(r'"parameters":\s*"([^"]+)"', input_string)
        if parameters_match:
            param_string = parameters_match.group(1).strip()
            param_pairs = re.findall(r'(\w+):\s*([\w\.]+)', param_string)
            for param_key, param_value in param_pairs:
                try:
                    param_value = int(param_value)
                except ValueError:
                    pass
                parameters[param_key] = param_value

        structured_data["parameters"] = parameters

        plane_match = re.search(r'"plane":\s*"([^"]+)"', input_string)
        if plane_match:
            structured_data['plane'] = plane_match.group(1).strip()

        coordinates_match = re.search(r'"coordinates":\s*(\[[^\]]+\])', input_string)
        if coordinates_match:
            coordinates = json.loads(coordinates_match.group(1).strip())
            structured_data["coordinates"] = coordinates

        return structured_data

    except Exception as e:
        print(f"Error in formatting: {e}")
        return None


HUGGINGFACE_TOKEN = "</>"

# Load and Split Dataset
def load_and_split_dataset(file_path):
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=file_path)
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    print("Dataset loaded and split successfully.")
    return split_dataset


def preprocess_function(example, tokenizer):
    return tokenizer(
        example["input"],
        text_target=example["output"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def fine_tune_model(dataset, tokenizer):
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", token=HUGGINGFACE_TOKEN)
    print("Model loaded successfully.")

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    print("Dataset tokenized successfully.")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./fine_tuned_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=50,
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        fp16=False
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")

    print("Saving fine-tuned model...")
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Model saved successfully.")
    return model
