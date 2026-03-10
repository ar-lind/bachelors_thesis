from dataset_seen_and_unseen import load_data
from unseen import unseen_load_data
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer,\
    DataCollatorForSeq2Seq
import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
import os
import yaml

with open(os.path.join("./configs", "config.yaml")) as f:
    config = yaml.safe_load(f)
tokenizer = AutoTokenizer.from_pretrained(config["model"])


def tokenize(tuples):
    """
    Tokenizes the tuples.
    Reference:
        https://huggingface.co/docs/evaluate/en/transformers_integrations
        Seq2SeqTrainer preprocess_function

    :param tuples: the tuples in the dataset
    :return: the tokenized tuples
    """
    tokenized_tuples = tokenizer(
        tuples["mod_transcription"],
        truncation=True,
        max_length=256
    )

    labels = tokenizer(
        text_target=tuples["transcription"],
        truncation=True,
        max_length=256
    )
    tokenized_tuples["labels"] = labels["input_ids"]

    return tokenized_tuples


def compute_metrics(eval_preds):
    """
    Computes the cer of the predictions.
    Reference:
        https://huggingface.co/docs/evaluate/en/transformers_integrations
        Seq2SeqTrainer compute_metrics

    :param eval_preds:
    :return:
    """
    cer = evaluate.load("cer")
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return {"cer": cer.compute(predictions=decoded_preds, references=decoded_labels)}


def save_preds(preds: np.ndarray | tuple[np.ndarray], metrics: dict[str, float] | None,
               test_dataset:  dict | Dataset | DatasetDict):
    """
    Saves the models predictions.
    Reference: AI generated (gpt-5)

    :param preds: predictions
    :param metrics: metrics
    :param test_dataset: the evaluation dataset
    """
    if isinstance(preds, (tuple, list)):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    noisy = test_dataset["mod_transcription"]
    clean = test_dataset["transcription"]

    with open("out/seen_preds.txt", "w", encoding="utf-8") as f:
        f.write(str(metrics) + "\n\n")
        for n, c, p in zip(noisy, clean, pred_texts):
            f.write(f"NOISY: {n}\nCLEAN: {c}\nPRED : {p}\n" + "-" * 60 + "\n")


def training(mode: str = "seen"):
    """
    Performs the training and evaluation via TrainerAPI.

    Reference:
        https://huggingface.co/docs/evaluate/en/transformers_integrations
        Seq2SeqTrainer
    """
    with open(os.path.join("./configs", "config.yaml")) as f:
        config = yaml.safe_load(f)

    if mode == "seen":
        train_dataset, test_dataset = load_data()
        train_dataset = train_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)
    else:
        train_dataset, test_dataset = unseen_load_data()
        train_dataset = train_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"])
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./out",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        save_safetensors=True,
        push_to_hub=False,
        load_best_model_at_end=False,
        metric_for_best_model="cer",
        greater_is_better=False,
        generation_max_length=256,
        predict_with_generate=True,
        num_train_epochs=10
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    preds = trainer.predict(
        test_dataset=test_dataset
    ).predictions
    metrics = trainer.predict(
        test_dataset=test_dataset
    ).metrics

    save_preds(preds, metrics, test_dataset)


if __name__ == '__main__':
    training(mode="seen")  # change to unseen
