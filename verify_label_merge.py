import sys
import os
import json
import glob
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TRAIN_LABELS_DIR, VAL_LABELS_DIR, FEATURES
from src.label_merge import merge_label, LABEL_MERGE_RULES, clean_label


def load_all_labels(labels_dir):
    all_labels = []
    json_files = glob.glob(os.path.join(labels_dir, "*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                label = json.load(f)
            all_labels.append(label)
        except Exception as e:
            pass
    return all_labels


def main():
    print("=" * 70)
    print("标签归并验证报告")
    print("=" * 70)

    train_labels = load_all_labels(TRAIN_LABELS_DIR)
    val_labels = load_all_labels(VAL_LABELS_DIR)
    all_labels = train_labels + val_labels

    print(f"\n总样本数: {len(all_labels)} (训练集: {len(train_labels)}, 验证集: {len(val_labels)})")

    print("\n" + "=" * 70)
    print("归并规则总览")
    print("=" * 70)
    for feature, rules in LABEL_MERGE_RULES.items():
        print(f"\n【{feature}】 归并为 {len(rules)} 个大类:")
        for merged_label, keywords in rules.items():
            print(f"  - {merged_label} (关键词: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''})")

    print("\n" + "=" * 70)
    print("归并前后类别数对比")
    print("=" * 70)
    print(f"{'特征':<20}{'归并前':<10}{'归并后':<10}{'减少':<10}{'减少比例':<10}")
    print("-" * 60)

    orig_counters = {}
    merged_counters = {}

    for feature in FEATURES:
        orig_values = []
        merged_values = []
        for label in all_labels:
            val = label.get(feature, "")
            if val and val.strip():
                cleaned = clean_label(val)
                if cleaned:
                    orig_values.append(cleaned)
                    merged = merge_label(feature, val)
                    if merged:
                        merged_values.append(merged)

        orig_counter = Counter(orig_values)
        merged_counter = Counter(merged_values)
        orig_counters[feature] = orig_counter
        merged_counters[feature] = merged_counter

        before = len(orig_counter)
        after = len(merged_counter)
        reduction = before - after
        ratio = reduction / before * 100 if before > 0 else 0
        print(f"{feature:<20}{before:<10}{after:<10}{reduction:<10}{ratio:.0f}%")

    print("\n" + "=" * 70)
    print("归并后各类别样本分布")
    print("=" * 70)

    for feature in FEATURES:
        counter = merged_counters[feature]
        total = sum(counter.values())
        print(f"\n【{feature}】 (共 {total} 个有效样本, {len(counter)} 个类别)")
        print(f"  {'类别':<15}{'样本数':<10}{'占比':<10}")
        print("  " + "-" * 35)
        for label, count in counter.most_common():
            pct = count / total * 100
            print(f"  {label:<15}{count:<10}{pct:.1f}%")

    print("\n" + "=" * 70)
    print("验证集标签覆盖率")
    print("=" * 70)

    for feature in FEATURES:
        train_merged = set()
        val_merged = set()
        for label in train_labels:
            val = label.get(feature, "")
            if val:
                merged = merge_label(feature, val)
                if merged:
                    train_merged.add(merged)
        for label in val_labels:
            val = label.get(feature, "")
            if val:
                merged = merge_label(feature, val)
                if merged:
                    val_merged.add(merged)

        uncovered = val_merged - train_merged
        if uncovered:
            print(f"\n【{feature}】 验证集有 {len(uncovered)} 个类别不在训练集中: {uncovered}")
        else:
            print(f"\n【{feature}】 验证集所有类别均在训练集中，覆盖率 100%")

    print("\n" + "=" * 70)
    print("完整性验证")
    print("=" * 70)

    all_ok = True
    for feature in FEATURES:
        counter = merged_counters[feature]
        if len(counter) < 2:
            print(f"  [警告] {feature} 类别数少于 2")
            all_ok = False
        min_samples = min(counter.values()) if counter else 0
        max_samples = max(counter.values()) if counter else 0
        print(f"  {feature}: {len(counter)} 类, 最少 {min_samples} 样本, 最多 {max_samples} 样本")

    if all_ok:
        print("\n  所有特征标签系统完整，可以用于训练。")
    else:
        print("\n  存在问题，请检查。")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
