"""
为 JSON 数据中 blocks 下 children 里的每个元素添加唯一 id。
"""
from typing import Any, Dict, List, Optional


def _add_children_ids_to_data(
    data: Any,
    *,
    id_template: Optional[str] = None,
    in_place: bool = True,
) -> Any:
    """
    支持多种输入形态：
    - list[slide]: 直接视为 slides
    - dict 且包含 key="slides" (list): 处理 data["slides"]
    - dict: 视为单个 slide
    """
    if isinstance(data, list):
        return add_children_ids(data, id_template=id_template, in_place=in_place)
    if isinstance(data, dict) and isinstance(data.get("slides"), list):
        filtered = add_children_ids(data["slides"], id_template=id_template, in_place=False)
        # 原地替换，确保 data["slides"] 的引用不变
        data["slides"][:] = filtered
        return data
    if isinstance(data, dict):
        add_children_ids_to_single_slide(data, id_template=id_template, in_place=in_place)
        return data
    raise TypeError(f"Unsupported JSON top-level type: {type(data).__name__}")


def add_children_ids(
    slides: List[Dict[str, Any]],
    *,
    id_template: Optional[str] = None,
    in_place: bool = True,
) -> List[Dict[str, Any]]:
    """
    为所有 block 的 children 中的元素添加 id。

    Args:
        slides: 顶层 slide 列表，每个 slide 含 blocks，每个 block 含 children。
        id_template: 可选。id 格式模板，可用占位符：
            {block_id} = block 的 id
            {index}    = 该 child 在 children 中的下标
            默认等价于 "b{block_id}_c{index}"
        in_place: 为 True 时直接修改原对象；为 False 时深拷贝后修改并返回。

    Returns:
        处理后的 slides（若 in_place 为 True 则与传入为同一引用）。
    """
    if not in_place:
        import copy
        slides = copy.deepcopy(slides)

    default_template = "b{block_id}_c{index}"

    processed: List[Dict[str, Any]] = []
    for slide in slides:
        blocks = slide.get("blocks") or []
        # 如果 blocks 数量为 0：跳过，不加入处理后的数据
        if len(blocks) == 0:
            continue
        for block in blocks:
            block_id = block.get("id", 0)
            children = block.get("children") or []
            template = id_template or default_template
            for i, child in enumerate(children):
                child["id"] = template.format(
                    block_id=block_id,
                    index=i,
                )
        processed.append(slide)

    if in_place:
        slides[:] = processed
        return slides
    return processed


def add_children_ids_to_single_slide(
    slide: Dict[str, Any],
    *,
    id_template: Optional[str] = None,
    in_place: bool = True,
) -> Dict[str, Any]:
    """
    为单个 slide 内所有 block 的 children 添加 id。
    id 默认格式: "b{block_id}_c{index}"（不含 slide_id，因单 slide 内已可区分）。
    """
    return add_children_ids(
        [slide],
        id_template=id_template or "b{block_id}_c{index}",
        in_place=in_place,
    )[0]


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/test.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _add_children_ids_to_data(data, in_place=True)

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已写入: {out_path}")
    else:
        preview = data[:1] if isinstance(data, list) else data
        print(json.dumps(preview, ensure_ascii=False, indent=2)[:2000])
