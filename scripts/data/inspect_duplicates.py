from pathlib import Path

from dataset_utils import load_techqa_dataframe


def context_filename(contexts):
    if not contexts:
        return "(không có context)"
    return ", ".join(context.get("filename", "?") for context in contexts)


def main():
    dataframe = load_techqa_dataframe()
    duplicates = dataframe[
        dataframe.duplicated(subset="question", keep=False)
    ].copy()

    if duplicates.empty:
        print("Không có câu hỏi trùng.")
        return

    duplicates["context_file"] = duplicates["contexts"].apply(context_filename)

    duplicate_groups = duplicates["question"].nunique()
    cross_split_groups = int(
        (duplicates.groupby("question")["source_split"].nunique() > 1).sum()
    )
    conflicting_label_groups = int(
        (duplicates.groupby("question")["is_impossible"].nunique() > 1).sum()
    )

    report_path = Path("reports/duplicate_questions.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    duplicates[
        ["id", "source_split", "is_impossible", "context_file", "question"]
    ].sort_values(["question", "id"]).to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("Số nhóm câu hỏi trùng:", duplicate_groups)
    print("Số nhóm trùng giữa TRAIN và DEV:", cross_split_groups)
    print("Số nhóm có nhãn trả lời trái ngược:", conflicting_label_groups)
    print("Chi tiết đã lưu tại:", report_path)


if __name__ == "__main__":
    main()