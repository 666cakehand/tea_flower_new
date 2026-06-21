import os
import json
import glob
from collections import Counter, defaultdict
from src.config import TRAIN_LABELS_DIR, VAL_LABELS_DIR, FEATURES


def load_all_labels(labels_dir):
    all_labels = []
    json_files = glob.glob(os.path.join(labels_dir, "*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                label = json.load(f)
            all_labels.append(label)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return all_labels


def analyze_feature(labels, feature):
    values = []
    for label in labels:
        val = label.get(feature, "")
        if val and val.strip():
            values.append(val.strip())

    counter = Counter(values)
    total = len(values)

    print(f"\n{'='*70}")
    print(f"【{feature}】 共 {total} 个有效标签，{len(counter)} 个不同类别")
    print(f"{'='*70}")

    print(f"\n所有类别按频率排序（前50个）:")
    print(f"{'排名':<6}{'标签值':<40}{'数量':<8}{'占比':<8}")
    print("-" * 70)
    for i, (val, count) in enumerate(counter.most_common(50), 1):
        pct = count / total * 100
        print(f"{i:<6}{val:<40}{count:<8}{pct:.1f}%")

    if len(counter) > 50:
        print(f"\n... 还有 {len(counter) - 50} 个类别")

    print(f"\n稀有类别统计:")
    print(f"  只有1个样本的类别（稀有类别）: {sum(1 for v, c in counter.items() if c == 1)} 个")
    print(f"  只有2个样本的类别: {sum(1 for v, c in counter.items() if c == 2)} 个")
    print(f"  样本数<=5的类别: {sum(1 for v, c in counter.items() if c <= 5)} 个")

    print(f"\n命名规范问题检查:")
    trailing_punct = [v for v in counter.keys() if v and v[-1] in '，。、,.;；:：  ']
    if trailing_punct:
        print(f"  末尾带标点/空格的标签: {len(trailing_punct)} 个")
        for v in trailing_punct[:10]:
            print(f"      '{v}'")

    has_extra_desc = [v for v in counter.keys() if len(v) > 15 and ('，' in v or '。' in v or ',' in v)]
    if has_extra_desc:
        print(f"  含额外描述的长标签: {len(has_extra_desc)} 个")
        for v in has_extra_desc[:10]:
            print(f"      '{v}'")

    return counter


def find_similar_groups(counter, feature):
    values = list(counter.keys())
    groups = defaultdict(list)

    for val in values:
        cleaned = val.strip().rstrip('，。、,.;；:：  ').rstrip('，。、,.;；:：  ')
        cleaned = cleaned.replace(' ', '')

        if feature == "花瓣外侧主色的颜色":
            for color_base in ["红色", "粉红色", "白色", "黄色", "橙色", "紫色", "绿色", "蓝色", "粉色", "紫红", "橙红", "玫红", "深红", "浅红", "桃红", "橘红", "朱红", "艳红", "鲜红", "淡红", "浅粉", "深粉", "淡粉", "粉底", "白瓣", "白花", "奶白", "米白", "雪白", "乳白", "淡黄", "金黄", "橙黄", "浅黄", "深黄", "紫", "蓝紫", "淡紫", "浅紫", "深紫", "绿", "浅绿", "深绿", "墨绿", "蓝", "天蓝", "淡蓝"]:
                if color_base in cleaned:
                    groups[color_base].append(val)
                    break

        elif feature == "花型":
            for type_base in ["单瓣", "重瓣", "半重瓣", "托桂", "绣球", "牡丹", "菊花", "蔷薇", "莲座", "杯状", "喇叭", "钟形", "碗状", "球状", "扁平", "十字", "星形"]:
                if type_base in cleaned:
                    groups[type_base].append(val)
                    break

    return groups


def main():
    print("加载训练集标签...")
    train_labels = load_all_labels(TRAIN_LABELS_DIR)
    print(f"  训练集: {len(train_labels)} 个样本")

    print("\n加载验证集标签...")
    val_labels = load_all_labels(VAL_LABELS_DIR)
    print(f"  验证集: {len(val_labels)} 个样本")

    all_labels = train_labels + val_labels
    print(f"\n  总计: {len(all_labels)} 个样本")

    counters = {}
    similar_groups = {}
    for feature in FEATURES:
        counters[feature] = analyze_feature(all_labels, feature)
        similar_groups[feature] = find_similar_groups(counters[feature], feature)

    print(f"\n\n{'='*70}")
    print("潜在相似标签组分析（可归并）")
    print(f"{'='*70}")

    for feature in FEATURES:
        groups = similar_groups[feature]
        print(f"\n【{feature}】 可归并为 {len(groups)} 个大类")
        mergeable = sum(len(vals) for vals in groups.values())
        original = len(counters[feature])
        print(f"  原类别数: {original}, 归并后约: {len(groups) + (original - mergeable)} (含无法归并的)")

        for base, vals in sorted(groups.items(), key=lambda x: -len(x[1])):
            if len(vals) >= 2:
                total_samples = sum(counters[feature][v] for v in vals)
                print(f"\n  【{base}】({len(vals)}个标签变体, {total_samples}个样本)")
                for v in vals[:8]:
                    print(f"    - '{v}' ({counters[feature][v]}个)")
                if len(vals) > 8:
                    print(f"    ... 还有 {len(vals)-8} 个")

    print(f"\n\n{'='*70}")
    print("归并前后预估对比")
    print(f"{'='*70}")
    print(f"{'特征':<20}{'归并前类别数':<15}{'归并后约':<15}{'减少比例':<10}")
    print("-" * 60)
    for feature in FEATURES:
        original = len(counters[feature])
        groups = similar_groups[feature]
        mergeable = sum(len(vals) for vals in groups.values())
        estimated = len(groups) + (original - mergeable)
        reduction = (1 - estimated / original) * 100
        print(f"{feature:<20}{original:<15}{estimated:<15}{reduction:.0f}%")


if __name__ == "__main__":
    main()
