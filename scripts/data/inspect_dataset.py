from pathlib import Path

import pandas as pd

from dataset_utils import load_techqa_dataframe


def main():
    dataframe = load_techqa_dataframe()
    context_counts = dataframe["contexts"].apply(
        lambda contexts: len(contexts) if isinstance(contexts, list) else 0
    )

    answerable_count = int((~dataframe["is_impossible"]).sum())
    impossible_count = int(dataframe["is_impossible"].sum())
    missing_by_column = dataframe.isna().sum()
    split_counts = dataframe["source_split"].value_counts(dropna=False)
    context_label_table = pd.crosstab(dataframe["is_impossible"], context_counts)

    print("Dataset: nvidia/TechQA-RAG-Eval")
    print("Số dòng:", len(dataframe))
    print("Các cột:", ", ".join(dataframe.columns.drop("source_split")))
    print("Số ô bị thiếu:", int(missing_by_column.sum()))
    print("Số câu trả lời được:", answerable_count)
    print("Số câu không thể trả lời:", impossible_count)
    print("Số dòng theo nguồn:", split_counts.to_dict())
    print("\nNhãn is_impossible so với số context:")
    print(context_label_table.to_string())

    report_lines = [
        "# TechQA dataset audit",
        "",
        f"- Số dòng: {len(dataframe)}",
        f"- Các cột: {', '.join(dataframe.columns.drop('source_split'))}",
        f"- Số ô bị thiếu: {int(missing_by_column.sum())}",
        f"- Câu trả lời được: {answerable_count}",
        f"- Câu không thể trả lời: {impossible_count}",
        "- Số dòng theo nguồn:",
    ]
    report_lines.extend(
        f"  - {split}: {count}" for split, count in split_counts.items()
    )
    report_lines.extend([
        "",
        "## Nhãn và số context",
        "",
        context_label_table.to_string(),
        "",
    ])

    report_path = Path("reports/dataset_audit.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\nBản tóm tắt đã lưu tại:", report_path)


if __name__ == "__main__":
    main()