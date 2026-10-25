import pandas as pd

from dataset_utils import get_all_contexts, load_techqa_dataframe


def main():
    dataframe = load_techqa_dataframe()
    contexts = get_all_contexts(dataframe)
    contexts_dataframe = pd.DataFrame(contexts)

    filename_counts = contexts_dataframe["filename"].value_counts()
    text_versions = contexts_dataframe.groupby("filename")["text"].nunique()
    repeated_filenames = filename_counts[filename_counts > 1]
    filenames_with_multiple_texts = text_versions[text_versions > 1]

    print("Tổng lượt context:", len(contexts))
    print("Số tài liệu khác nhau:", len(filename_counts))
    print("Số tài liệu xuất hiện nhiều lần:", len(repeated_filenames))
    print("Tên file có nhiều phiên bản nội dung:", len(filenames_with_multiple_texts))

    if not repeated_filenames.empty:
        print("\n10 tài liệu được dùng nhiều nhất:")
        print(repeated_filenames.head(10).to_string())


if __name__ == "__main__":
    main()