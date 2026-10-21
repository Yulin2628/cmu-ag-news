from datasets import load_dataset

SEED = 42
DATASET_REVISION = 'eb185aade064a813bc0b7f42de02595523103ca4'

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def prepare_data():
    raw = load_dataset("fancyzhx/ag_news", revision=DATASET_REVISION)

    # Remove exact duplicate text before splitting. Exclude official test texts
    # from the training pool; test labels never affect model/config selection.
    def unique_texts(data, excluded=None):
        seen = set() if excluded is None else set(excluded)
        keep = []
        for i, text in enumerate(data["text"]):
            key = text.strip()
            if key not in seen:
                keep.append(i)
                seen.add(key)
        return data.select(keep)

    raw["test"] = unique_texts(raw["test"])
    raw["train"] = unique_texts(raw["train"], raw["test"]["text"])

    selected = raw["train"].train_test_split(
        test_size=2400,
        stratify_by_column="label",
        seed=SEED,
    )["test"]

    train_dev = selected.train_test_split(
        test_size=400,
        stratify_by_column="label",
        seed=SEED,
    )

    train_data = train_dev["train"]
    dev_data = train_dev["test"]

    # Reconstruct the pilot holdout so it can be excluded from the new test pool.
    previously_used = raw["test"].train_test_split(
        test_size=400,
        stratify_by_column="label",
        seed=SEED,
    )["test"]
    excluded = set(previously_used["text"])
    fresh_test_pool = raw["test"].filter(lambda row: row["text"] not in excluded)
    test_data = fresh_test_pool.train_test_split(
        test_size=400, stratify_by_column="label", seed=43,
    )["test"]

    return train_data, dev_data, test_data


if __name__ == "__main__":
    train_data, dev_data, test_data = prepare_data()

    print("训练集数量：", len(train_data))
    print("验证集数量：", len(dev_data))
    print("测试集数量：", len(test_data))

    example = train_data[0]

    print("\n一条训练样本：")
    print("新闻：", example["text"])
    print("类别编号：", example["label"])
    print("类别名称：", LABEL_NAMES[example["label"]])
