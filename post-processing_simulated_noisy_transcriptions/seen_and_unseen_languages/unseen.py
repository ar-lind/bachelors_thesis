import pandas as pd
from datasets import load_dataset, Dataset
import random
import yaml
import os


def preprocessing():
    book_df = pd.read_csv("path/to/book/folder/examples.csv")  # add loaded source folder with book examples
    book_df = book_df[book_df["Language_ID"] == "kara1499"]
    book_df = book_df[["Primary_Text"]]
    book_df = book_df.rename(columns={"Primary_Text": "transcription"})
    book_df = book_df.sort_values("transcription")
    book_df["transcription"] = book_df["transcription"].map(str.lower)
    book_df["transcription"] = book_df["transcription"].map(remove_chars)

    test_ds = load_dataset("wav2gloss/fieldwork", split="test").select_columns(["id", "language", "transcription"])
    test_df = test_ds.to_pandas()
    test_df = test_df[test_df["language"] == "kara1499"]
    test_df = test_df[test_df["transcription"] != ""]
    test_df = test_df[["transcription"]]
    test_df = test_df.sort_values("transcription")

    val_ds = load_dataset("wav2gloss/fieldwork", split="validation").select_columns(["id", "language", "transcription"])
    val_df = val_ds.to_pandas()
    val_df = val_df[val_df["language"] == "kara1499"]
    val_df = val_df[val_df["transcription"] != ""]
    val_df = val_df[["transcription"]]
    val_df = val_df.sort_values("transcription")

    fieldwork_df = pd.concat([test_df, val_df])

    res = book_df[~book_df["transcription"].isin(fieldwork_df["transcription"])]
    return res


def remove_chars(transcription: str):
    res_str = ""
    for c in transcription:
        if c not in ["=", "-", "∅", "“", "”", "[", "]", "∼", "\""]:
            res_str += c
    return res_str


def unseen_load_data():
    """
    Loads the preprocessed transcriptions and adds noise.
    Reference: https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html

    :returns: train_dataset, test_dataset: tuple of training dataset and testing dataset
    """

    train_transcriptions_df = preprocessing()
    train_transcriptions_df_base = pd.concat([train_transcriptions_df, train_transcriptions_df])
    train_transcriptions_df_base["mod_transcription"] = train_transcriptions_df_base["transcription"].map(add_base_noise)
    train_transcriptions_df_high = train_transcriptions_df.copy()
    train_transcriptions_df_high["mod_transcription"] = train_transcriptions_df_high["transcription"].map(add_high_noise)
    train_transcriptions_df_low = train_transcriptions_df.copy()
    train_transcriptions_df_low["mod_transcription"] = train_transcriptions_df_low["transcription"].map(add_low_noise)
    train_transcriptions_df = pd.concat([train_transcriptions_df_base, train_transcriptions_df_high, train_transcriptions_df_low])

    new_train_df = train_transcriptions_df.drop_duplicates()

    test_transcriptions_ds = load_dataset("wav2gloss/fieldwork", split="test")\
        .select_columns(["transcription", "language"])
    test_transcriptions_df = test_transcriptions_ds.to_pandas()
    test_transcriptions_df = test_transcriptions_df[test_transcriptions_df["language"] == "kara1499"]
    test_transcriptions_df = test_transcriptions_df[test_transcriptions_df["transcription"] != ""]
    test_transcriptions_df["mod_transcription"] = test_transcriptions_df["transcription"].map(add_base_noise)

    new_test_df = test_transcriptions_df.drop_duplicates()

    train_dataset = Dataset.from_pandas(new_train_df)

    test_dataset = Dataset.from_pandas(new_test_df)

    return train_dataset.shuffle(), test_dataset.shuffle()


def add_base_noise(transcription: str):
    """
    Adds base noise to the original string.

    :param transcription: the original transcription string without noise

    :return: the modified transcription string with noise
    """

    with open(os.path.join("./configs", "config.yaml")) as f:
        config = yaml.safe_load(f)

    original_transcription = transcription

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < config["unseen_del"]:
            if len(transcription) < 1:
                continue
            del_index = random.randint(0, len(transcription)-1)
            transcription = transcription[:del_index] + transcription[del_index+1:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < config["unseen_ins"]:
            if len(transcription) < 1:
                continue
            ins_index = random.randint(0, len(transcription)-1)

            ins_char = transcription[random.randint(0, len(transcription)-1)]

            transcription = transcription[:ins_index] + ins_char + transcription[ins_index:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < config["unseen_sub"]:
            if len(transcription) < 1:
                continue
            sub_index = random.randint(0, len(transcription)-1)

            sub_char = transcription[random.randint(0, len(transcription)-1)]

            transcription = transcription[:sub_index] + sub_char + transcription[sub_index+1:]

    return transcription


def add_high_noise(transcription: str):
    """
    Adds high noise to the original string.

    :param transcription: the original transcription string without noise

    :return: the modified transcription string with noise
    """

    with open(os.path.join("./configs", "config.yaml")) as f:
        config = yaml.safe_load(f)

    original_transcription = transcription
    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < (config["unseen_del"] + 100):
            if len(transcription) < 1:
                continue
            del_index = random.randint(0, len(transcription)-1)
            transcription = transcription[:del_index] + transcription[del_index+1:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < (config["unseen_ins"] + 100):
            if len(transcription) < 1:
                continue
            ins_index = random.randint(0, len(transcription)-1)

            ins_char = transcription[random.randint(0, len(transcription)-1)]

            transcription = transcription[:ins_index] + ins_char + transcription[ins_index:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < (config["unseen_sub"] + 100):
            if len(transcription) < 1:
                continue
            sub_index = random.randint(0, len(transcription)-1)

            sub_char = transcription[random.randint(0, len(transcription)-1)]

            transcription = transcription[:sub_index] + sub_char + transcription[sub_index+1:]

    return transcription


def add_low_noise(transcription: str):
    """
    Adds low noise to the original string.

    :param transcription: the original transcription string without noise

    :return: the modified transcription string with noise
    """

    with open(os.path.join("./configs", "config.yaml")) as f:
        config = yaml.safe_load(f)

    original_transcription = transcription

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < int(config["unseen_del"] / 2):
            if len(transcription) < 1:
                continue
            del_index = random.randint(0, len(transcription)-1)
            transcription = transcription[:del_index] + transcription[del_index+1:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < int(config["unseen_ins"] / 2):
            if len(transcription) < 1:
                continue
            ins_index = random.randint(0, len(transcription)-1)

            ins_char = transcription[random.randint(0, len(transcription)-1)]

            transcription = transcription[:ins_index] + ins_char + transcription[ins_index:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < int(config["unseen_sub"] / 2):
            if len(transcription) < 1:
                continue
            sub_index = random.randint(0, len(transcription)-1)

            sub_char = transcription[random.randint(0, len(transcription)-1)]

            transcription = transcription[:sub_index] + sub_char + transcription[sub_index+1:]

    return transcription


if __name__ == '__main__':
    training_dataset, testing_dataset = unseen_load_data()
    print(len(training_dataset))
