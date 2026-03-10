import random
from datasets import load_dataset, Dataset
import pandas as pd
import yaml
import os


def load_data():
    """
    Loads the transcriptions from the corpus and adds noise.
    Reference: https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html

    :returns: train_dataset, test_dataset: tuple of training dataset and testing dataset
    """

    with open(os.path.join("./configs", "config.yaml")) as f:
        config = yaml.safe_load(f)

    train_transcriptions_ds = load_dataset("wav2gloss/fieldwork", split="train")\
        .select_columns(["transcription", "language"])
    train_transcriptions_df = train_transcriptions_ds.to_pandas()
    train_transcriptions_df = train_transcriptions_df[train_transcriptions_df["language"] == config["language"]]
    train_transcriptions_df_base = pd.concat([train_transcriptions_df, train_transcriptions_df])
    train_transcriptions_df_base["mod_transcription"] = train_transcriptions_df_base["transcription"].map(add_base_noise)
    train_transcriptions_df_high = train_transcriptions_df.copy()
    train_transcriptions_df_high["mod_transcription"] = train_transcriptions_df_high["transcription"].map(add_high_noise)
    train_transcriptions_df_low = train_transcriptions_df.copy()
    train_transcriptions_df_low["mod_transcription"] = train_transcriptions_df_low["transcription"].map(add_low_noise)
    train_transcriptions_df = \
        pd.concat([train_transcriptions_df_base, train_transcriptions_df_high, train_transcriptions_df_low])
    train_transcriptions_df = train_transcriptions_df[train_transcriptions_df["transcription"] != ""]

    new_train_df = train_transcriptions_df.drop_duplicates()

    test_df = pd.read_csv("sources/" + config["language"] + "_test_inference.csv")
    test_df = test_df[["ref", "hyp"]].rename(columns={
        "ref": "transcription",
        "hyp": "mod_transcription"
    })
    test_df["transcription"] = test_df["transcription"].astype(str)
    test_df["mod_transcription"] = test_df["mod_transcription"].astype(str)
    test_df = test_df[test_df["transcription"] != ""]
    new_test_df = test_df[test_df["mod_transcription"] != ""]

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
        if random.randint(0, 1000) < config["del"]:
            if len(transcription) < 1:
                continue
            del_index = random.randint(0, len(transcription) - 1)
            transcription = transcription[:del_index] + transcription[del_index + 1:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < config["ins"]:
            if len(transcription) < 1:
                continue
            ins_index = random.randint(0, len(transcription) - 1)

            ins_char = transcription[random.randint(0, len(transcription) - 1)]

            transcription = transcription[:ins_index] + ins_char + transcription[ins_index:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < config["sub"]:
            if len(transcription) < 1:
                continue
            sub_index = random.randint(0, len(transcription) - 1)

            sub_char = transcription[random.randint(0, len(transcription) - 1)]

            transcription = transcription[:sub_index] + sub_char + transcription[sub_index + 1:]

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
        if random.randint(0, 1000) < (config["del"] + 100):
            if len(transcription) < 1:
                continue
            del_index = random.randint(0, len(transcription) - 1)
            transcription = transcription[:del_index] + transcription[del_index + 1:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < (config["ins"] + 100):
            if len(transcription) < 1:
                continue
            ins_index = random.randint(0, len(transcription) - 1)

            ins_char = transcription[random.randint(0, len(transcription) - 1)]

            transcription = transcription[:ins_index] + ins_char + transcription[ins_index:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < (config["sub"] + 100):
            if len(transcription) < 1:
                continue
            sub_index = random.randint(0, len(transcription) - 1)

            sub_char = transcription[random.randint(0, len(transcription) - 1)]

            transcription = transcription[:sub_index] + sub_char + transcription[sub_index + 1:]

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
        if random.randint(0, 1000) < int(config["del"] / 2):
            if len(transcription) < 1:
                continue
            del_index = random.randint(0, len(transcription) - 1)
            transcription = transcription[:del_index] + transcription[del_index + 1:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < int(config["ins"] / 2):
            if len(transcription) < 1:
                continue
            ins_index = random.randint(0, len(transcription) - 1)

            ins_char = transcription[random.randint(0, len(transcription) - 1)]

            transcription = transcription[:ins_index] + ins_char + transcription[ins_index:]

    for i in range(0, len(original_transcription)):
        if random.randint(0, 1000) < int(config["sub"] / 2):
            if len(transcription) < 1:
                continue
            sub_index = random.randint(0, len(transcription) - 1)

            sub_char = transcription[random.randint(0, len(transcription) - 1)]

            transcription = transcription[:sub_index] + sub_char + transcription[sub_index + 1:]

    return transcription


if __name__ == '__main__':
    training_dataset, testing_dataset = load_data()
    print(len(training_dataset))
