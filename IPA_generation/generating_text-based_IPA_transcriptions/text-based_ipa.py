import io
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import yaml
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import librosa
import numpy as np
import evaluate
from phonemizer import phonemize
import uroman as ur


def load_data():
    """
    Loads the transcriptions from the corpus and adds ipa column.
    Reference: https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html

    :returns: train_dataset, test_dataset: tuple of training dataset and testing dataset
    """

    # add loaded source folder with text examples
    train_df = pd.read_csv("path/to/fas_wikipedia_2014_10K-sentences.txt", sep="\t", names=["id", "transcription"])
    train_df = train_df[train_df["transcription"] != ""]
    transcriptions = train_df["transcription"].map(romanize).tolist()
    train_df["transcription"] = transcriptions
    ipa_transcriptions = phonemize(transcriptions, language="fa-latn", backend="espeak")
    train_df["ipa_transcription"] = ipa_transcriptions

    test_ds = load_dataset("wav2gloss/fieldwork", split="test").select_columns(["transcription", "language", "audio"])
    test_df = test_ds.to_pandas()
    test_df = test_df[test_df["language"] == config["language"]]
    test_df = test_df[test_df["transcription"] != ""]
    test_df["ipa_transcription"] = test_df["audio"].map(get_ipa)

    train_df.to_parquet(config["language"] + "_train_df.parquet")
    train_dataset = Dataset.from_pandas(train_df)

    test_df.to_parquet(config["language"] + "_test_df.parquet")
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset.shuffle(), test_dataset.shuffle()


processor = WhisperProcessor.from_pretrained("neurlang/ipa-whisper-base")
ipa_model = WhisperForConditionalGeneration.from_pretrained("neurlang/ipa-whisper-base")
ipa_model.config.forced_decoder_ids = None
ipa_model.config.suppress_tokens = []
ipa_model.generation_config.forced_decoder_ids = None
ipa_model.generation_config._from_model_config = True
c = 0


def get_ipa(audio: dict):
    """
    Generates ipa of the original string.
    Reference: https://huggingface.co/neurlang/ipa-whisper-base

    :param audio: the original audio

    :return: the ipa transcription string
    """
    global c
    c += 1
    print(c)
    sample_array, sample_sr = librosa.load(io.BytesIO(audio["bytes"]), sr=16000)
    input_features = processor(sample_array, sampling_rate=sample_sr, return_tensors="pt").input_features
    predicted_ids = ipa_model.generate(
        input_features=input_features,
        max_new_tokens=256,
        no_repeat_ngram_size=4
    )
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]


uroman = ur.Uroman()


def romanize(transcription: str):
    """

    :param transcription:
    :return:
    """
    global uroman
    transcription = uroman.romanize_string(transcription, lcode="fas")

    return transcription


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
        tuples["ipa_transcription"],
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
    pred_cnt = 0
    for t in pred_texts:
        pred_cnt += len(t.split())
    pred_avg = pred_cnt/len(pred_texts)

    ipa = test_dataset["ipa_transcription"]
    ipa_cnt = 0
    for t in ipa:
        ipa_cnt += len(t.split())
    ipa_avg = ipa_cnt/len(ipa)
    original = test_dataset["transcription"]
    original_cnt = 0
    for t in original:
        original_cnt += len(t.split())
    original_avg = original_cnt/len(original)

    with open("out/" + config["language"] + "_text_based_ipa_preds.txt", "w", encoding="utf-8") as g:
        g.write(str(metrics) + "\n\n")
        g.write("Average words in ipa: " + str(ipa_avg) + "\n")
        g.write("Average words in original: " + str(original_avg) + "\n")
        g.write("Average words in pred: " + str(pred_avg) + "\n\n")
        for i, o, p in zip(ipa, original, pred_texts):
            g.write(f"IPA: {i}\nORIGINAL: {o}\nPRED : {p}\n" + "-" * 60 + "\n")


def training():
    """
    Performs the training and evaluation via TrainerAPI.

    Reference:
        https://huggingface.co/docs/evaluate/en/transformers_integrations
        Seq2SeqTrainer

        https://huggingface.co/docs/transformers/main_classes/text_generation
        Generation
    """

    if os.path.exists(config["language"] + "_train_df.parquet"):
        train_dataset = pd.read_parquet(config["language"] + "_train_df.parquet")
        train_dataset = Dataset.from_pandas(train_dataset).map(tokenize, batched=True)
        test_dataset = pd.read_parquet(config["language"] + "_test_df.parquet")
        test_dataset = Dataset.from_pandas(test_dataset).map(tokenize, batched=True)
    else:
        train_dataset, test_dataset = load_data()
        train_dataset = train_dataset.map(tokenize, batched=True)
        test_dataset = test_dataset.map(tokenize, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"])
    model.generation_config.no_repeat_ngram_size = 4
    model.generation_config.repetition_penalty = 1.2
    model.generation_config.length_penalty = 1.0
    model.generation_config.early_stopping = True
    model.generation_config.num_beams = 4
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

    out = trainer.predict(
        test_dataset=test_dataset
    )
    preds = out.predictions
    metrics = out.metrics

    save_preds(preds, metrics, test_dataset)


if __name__ == '__main__':
    training()
