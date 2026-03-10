from datasets import Dataset
import pandas as pd
import yaml
import os

lang = "empty"


def load_data():
    """
    Loads the transcriptions from the corpus and adds noise.
    Reference: https://pandas.pydata.org/docs/getting_started/comparison/comparison_with_sql.html

    :returns: train_dataset, test_dataset: tuple of training dataset and testing dataset
    """

    with open(os.path.join("./configs", "config.yaml")) as f:
        config = yaml.safe_load(f)

    non_target_langs = ["beja1238", "even1259", "ruul1235", "teop1238", "selk1253", "slav1254", "sumi1235", "goro1270",
                        "komn1238"]
    non_target_langs.remove(config["language"])

    non_target_size = 0
    for non_target_lang in non_target_langs:
        non_target_size += len(pd.read_csv("sources/train_" + non_target_lang + "_pred_text",
                                           sep=r"<task\|transcription><lang\|" + non_target_lang + ">",
                                           names=["id", "mod_transcription"], engine="python"))

    target_size = len(pd.read_csv("sources/train_" + config["language"] + "_pred_text",
                                  sep=r"<task\|transcription><lang\|" + config["language"] + ">",
                                  names=["id", "mod_transcription"], engine="python"))
    concat_num = round(non_target_size/target_size)

    langs = non_target_langs.copy()
    while concat_num:
        langs.append(config["language"])
        concat_num -= 1

    global lang
    new_train_df = pd.DataFrame()
    for lang in langs:
        train_df_1 = pd.read_csv("sources/train_" + lang + "_pred_text",
                                 sep=r"<task\|transcription><lang\|" + lang + ">",
                                 names=["id", "mod_transcription"], engine="python")
        train_df_1["mod_transcription"] = train_df_1["mod_transcription"].astype(str).str.strip()
        train_df_1["mod_transcription"] = train_df_1["mod_transcription"].map(tagger)
        train_df_2 = pd.read_csv("sources/train_" + lang + "_original_text", sep="@", engine="python",
                                 header=None, names=["line"])
        lines = train_df_2["line"].map(splitter).tolist()
        train_df_2[["id", "transcription"]] = lines
        train_df_2 = train_df_2[["id", "transcription"]]
        train_df_1["id"] = train_df_1["id"].astype(str).str.strip()
        train_df_2["id"] = train_df_2["id"].astype(str).str.strip()

        train_df = pd.merge(train_df_1, train_df_2, on="id")
        train_df["transcription"] = train_df["transcription"].astype(str)
        train_df["mod_transcription"] = train_df["mod_transcription"].astype(str)
        train_df = train_df[train_df["transcription"] != ""]
        train_df = train_df[train_df["mod_transcription"] != ""]
        new_train_df = pd.concat([new_train_df, train_df])

    lang = config["language"]
    test_df_1 = pd.read_csv("sources/" + config["language"] + "_pred_text",
                            sep=r"<task\|transcription><lang\|" + lang + ">",
                            names=["id", "mod_transcription"], engine="python")
    test_df_1["mod_transcription"] = test_df_1["mod_transcription"].map(tagger)
    test_df_2 = pd.read_csv("sources/" + config["language"] + "_original_text", sep="@", engine="python",
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


def tagger(transcription: str):
    global lang
    return "<lang|" + lang + "> " + transcription


def splitter(line: str):
    split = line.split(" ", 1)
    return split[0], split[1]


if __name__ == '__main__':
    training_dataset, testing_dataset = load_data()
    print(len(training_dataset))
