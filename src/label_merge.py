import re

LABEL_MERGE_RULES = {
    "花型": {
        "单瓣型": ["单瓣"],
        "半重瓣型": ["半重瓣"],
        "重瓣型": ["重瓣", "玫瑰重瓣", "完全重瓣"],
        "牡丹型": ["牡丹"],
        "托桂型": ["托桂"],
        "绣球型": ["绣球"],
        "菊花型": ["菊花"],
        "蔷薇型": ["蔷薇"],
    },
    "花瓣皱褶": {
        "无或弱褶皱": ["无", "弱褶皱", "弱皱褶", "微皱褶", "略皱褶", "略褶皱", "略呈波浪状", "微波浪状", "略波浪状"],
        "中褶皱": ["中褶皱", "中皱褶", "中度褶皱"],
        "强褶皱": ["强褶皱", "强皱褶", "强烈褶皱", "非常皱褶", "瓣面皱褶", "瓣面褶皱"],
        "波浪/扭曲状": ["波浪", "扭曲", "内扣", "外翻", "内卷", "边缘", "锯齿"],
    },
    "花瓣外侧主色的颜色": {
        "红色系": ["红色", "大红", "深红", "暗红", "黑红", "酒红", "橙红", "橘红", "桃红", "朱红", "艳红", "鲜红", "枣红", "火红", "玫瑰红", "玫红", "胭脂红"],
        "粉红色系": ["粉红色", "粉色", "粉红", "淡粉", "浅粉", "深粉", "桃粉", "玫粉", "杏粉", "胭脂粉", "嫩粉", "水粉"],
        "白色系": ["白色", "乳白", "奶白", "米白", "雪白", "纯白", "银白", "粉白"],
        "黄色系": ["黄色", "淡黄", "浅黄", "金黄", "橙黄", "米黄", "嫩黄", "柠檬黄", "奶黄"],
        "橙色系": ["橙色", "橘色", "橙橘", "杏色"],
        "紫色系": ["紫色", "紫红", "淡紫", "浅紫", "深紫", "蓝紫", "堇紫", "茄色"],
        "绿色系": ["绿色", "浅绿", "淡绿", "翠绿", "嫩绿"],
        "复色系": ["复色", "白斑", "白边", "白条纹", "斑块", "云状", "混色", "镶边", "渐变", "双色", "三色"],
    },
    "花瓣着色类型": {
        "单色": ["单色"],
        "复色": ["复色", "双色", "三色", "混色"],
        "渐变型": ["渐变"],
        "斑点条纹型": ["斑点", "条纹", "白斑", "斑块", "云状", "镶边", "隐斑"],
    },
}


def clean_label(label):
    if not label:
        return ""
    cleaned = label.strip()
    cleaned = cleaned.strip("，。、,.;；:： 　")
    cleaned = re.sub(r"[，。、,.;；:：]+$", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def merge_label(feature, label_value):
    if not label_value or not label_value.strip():
        return ""

    cleaned = clean_label(label_value)
    if not cleaned:
        return ""

    rules = LABEL_MERGE_RULES.get(feature, {})

    for merged_label, keywords in rules.items():
        for kw in keywords:
            if kw in cleaned:
                return merged_label

    return cleaned


def get_merged_feature_labels(feature, original_labels):
    merged_map = {}
    for orig_label in original_labels:
        merged = merge_label(feature, orig_label)
        if merged:
            merged_map[orig_label] = merged
    return merged_map
