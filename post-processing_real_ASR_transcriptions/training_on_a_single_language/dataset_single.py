from datasets import Dataset
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

    train_df_1 = pd.read_csv("sources/train_" + config["language"] + "_pred_text",
                             sep=r"<task\|transcription><lang\|" + config["language"] + ">",
                             names=["id", "mod_transcription"], engine="python")
    train_df_2 = pd.read_csv("sources/train_" + config["language"] + "_original_text", sep="ü", engine="python",
                            header=None, names=["line"])
    lines = train_df_2["line"].map(splitter).tolist()
    train_df_2[["id", "transcription"]] = lines
    train_df_2 = train_df_2[["id", "transcription"]]
    train_df_1["id"] = train_df_1["id"].astype(str).str.strip()
    train_df_2["id"] = train_df_2["id"].astype(str).str.strip()
    train_df = pd.merge(train_df_1, train_df_2, on="id").drop_duplicates()
    train_df["transcription"] = train_df["transcription"].astype(str)
    train_df["mod_transcription"] = train_df["mod_transcription"].astype(str)
    train_df = train_df[train_df["transcription"] != ""]
    new_train_df = train_df[train_df["mod_transcription"] != ""]

    test_df_1 = pd.read_csv("sources/" + config["language"] + "_pred_text",
                            sep=r"<task\|transcription><lang\|" + config["language"] + ">",
                            names=["id", "mod_transcription"], engine="python")
    test_df_2 = pd.read_csv("sources/" + config["language"] + "_original_text", sep="ü", engine="python",
                            header=None, names=["line"])
    lines = test_df_2["line"].map(splitter).tolist()
    test_df_2[["id", "transcription"]] = lines
    test_df_2 = test_df_2[["id", "transcription"]]
    test_df_1["id"] = test_df_1["id"].astype(str).str.strip()
    test_df_2["id"] = test_df_2["id"].astype(str).str.strip()
    test_df = pd.merge(test_df_1, test_df_2, on="id").drop_duplicates()
    test_df["transcription"] = test_df["transcription"].astype(str)
    test_df["mod_transcription"] = test_df["mod_transcription"].astype(str)
    test_df = test_df[test_df["transcription"] != ""]
    new_test_df = test_df[test_df["mod_transcription"] != ""]

    train_dataset = Dataset.from_pandas(new_train_df)

    test_dataset = Dataset.from_pandas(new_test_df)

    return train_dataset.shuffle(), test_dataset.shuffle()


def splitter(line: str):
    split = line.split(" ", 1)
    return split[0], split[1]


if __name__ == '__main__':
    training_dataset, testing_dataset = load_data()
    print(len(training_dataset))
