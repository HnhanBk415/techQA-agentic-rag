from datasets import load_dataset
import pandas as pd


def load_techqa_dataframe():
    dataset = load_dataset(
        "nvidia/TechQA-RAG-Eval",
        split="train"
    )

    dataframe = pd.DataFrame(dataset)

    # Xem ID bắt đầu bằng TRAIN_ hay DEV_
    source_splits = []

    for record_id in dataframe["id"]:
        if record_id.startswith("TRAIN_"):
            source_splits.append("TRAIN")
        elif record_id.startswith("DEV_"):
            source_splits.append("DEV")
        else:
            source_splits.append("UNKNOWN")

    dataframe["source_split"] = source_splits

    return dataframe


def get_all_contexts(dataframe):
    all_contexts = []

    # Mỗi dòng có thể có một danh sách context
    for context_list in dataframe["contexts"]:
        if isinstance(context_list, list):
            for context in context_list:
                all_contexts.append(context)

    return all_contexts