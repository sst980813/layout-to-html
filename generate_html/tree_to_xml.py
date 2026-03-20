"""
将 slide JSON 解析为结构化 XML（用于提示大模型理解版式结构）。

设计目标：
1) 不输出绝对坐标与宽高（x/y/width/height）。
2) 仅依赖 binary_tree / full_binary_tree 构造布局层次（region tree）。
3) 将 blocks/children 的语义内容挂载到 region 中，形成“类似 HTML 结构”的 XML。

说明：
- 不同数据集对 binary_tree/full_binary_tree 的编码语义可能不同。
- 这里采用“结构优先”的通用解析：先把 full_binary_tree 转为 region 层次，
  再按顺序将 block 映射到叶子 region（不足则轮询）。
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RegionNode:
    name: str
    direction: str = "column"
    children: List["RegionNode"] = field(default_factory=list)

    def add(self, node: "RegionNode") -> None:
        self.children.append(node)


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _tokenize_tree_code(code: str) -> List[str]:
    return [c for c in _to_text(code).upper() if c in {"C", "L", "R"}]


def _build_region_tree_from_full_code(full_code: str) -> RegionNode:
    """
    基于 full_binary_tree 构建 region 树（启发式）：
    - C: 当前节点标记为容器（container）
    - L/R: 创建一个子 region 并下潜
    例如 token 序列会得到一棵可嵌套的区域树，便于后续映射 block。
    """
    tokens = _tokenize_tree_code(full_code)
    root = RegionNode(name="region_root", direction="column")
    if not tokens:
        return root

    stack: List[RegionNode] = [root]
    child_idx = 0
    last_was_lr = False

    for t in tokens:
        cur = stack[-1]
        if t == "C":
            # 容器节点倾向纵向分组，保留现有方向即可
            last_was_lr = False
            continue

        side = "left" if t == "L" else "right"
        child = RegionNode(
            name=f"region_{side}_{child_idx}",
            direction="row" if side == "left" else "column",
        )
        child_idx += 1
        cur.add(child)

        # 连续 L/R 时下潜；若出现 C 或分支切换，后续会在更上层继续展开
        if last_was_lr:
            # 防止树过深：仅在当前深度较浅时继续下潜
            if len(stack) < 6:
                stack.append(child)
        else:
            stack.append(child)
        last_was_lr = True

        # 适度回退，避免单链
        if len(stack) > 2 and len(stack[-2].children) >= 2:
            stack.pop()

    return root


def _collect_leaf_regions(node: RegionNode) -> List[RegionNode]:
    if not node.children:
        return [node]
    leaves: List[RegionNode] = []
    for c in node.children:
        leaves.extend(_collect_leaf_regions(c))
    return leaves


def _block_semantic(block: Dict[str, Any]) -> str:
    t = _to_text(block.get("type")).lower()
    if t:
        return t
    if block.get("children"):
        return "group"
    if _to_text(block.get("text_content")):
        return "text"
    return "unknown"


def _append_block_xml(parent: ET.Element, block: Dict[str, Any]) -> None:
    children = block.get("children") or []
    block_el = ET.SubElement(
        parent,
        "block",
        {
            "id": _to_text(block.get("id")) or "unknown",
            "semantic": _block_semantic(block),
        },
    )

    # 规则：若 block 含 children，则 block 自身不渲染，只渲染 children。
    # 因此仅在“无 children”时才保留 block 级 text_content。
    text_content = _to_text(block.get("text_content"))
    if text_content and not children:
        text_el = ET.SubElement(block_el, "text")
        text_el.text = text_content

    if not children:
        return

    children_el = ET.SubElement(block_el, "children")
    for ch in children:
        kind = _to_text(ch.get("kind")).lower() or "unknown"
        attrs = {"id": _to_text(ch.get("id")) or "", "kind": kind}
        if kind == "text":
            attrs["semantic_level"] = _to_text(ch.get("semantic_level")) or "0"
        node = ET.SubElement(children_el, "node", attrs)
        if kind == "text":
            node.text = _to_text(ch.get("content"))
        elif kind == "image":
            node.set("src", _to_text(ch.get("image_path")))


def _region_to_xml(node: RegionNode, parent: ET.Element) -> ET.Element:
    el = ET.SubElement(
        parent,
        "region",
        {"name": node.name, "direction": node.direction},
    )
    for c in node.children:
        _region_to_xml(c, el)
    return el


def slide_to_xml(slide: Dict[str, Any]) -> str:
    slide_id = _to_text(slide.get("slide_id")) or "unknown_slide"
    binary_tree = _to_text(slide.get("binary_tree"))
    full_binary_tree = _to_text(slide.get("full_binary_tree"))
    blocks = slide.get("blocks") or []

    root = ET.Element(
        "slide",
        {
            "id": slide_id,
            "layout_category": _to_text(slide.get("layout_category")) or "unknown",
            "binary_tree": binary_tree,
            "full_binary_tree": full_binary_tree,
        },
    )

    layout = ET.SubElement(root, "layout")
    region_tree = _build_region_tree_from_full_code(full_binary_tree or binary_tree)
    region_root_el = _region_to_xml(region_tree, layout)

    leaf_regions = _collect_leaf_regions(region_tree)
    if not leaf_regions:
        leaf_regions = [region_tree]

    # 建立 node->xml element 索引，便于把 block 写到对应 region
    node_to_xml: Dict[str, ET.Element] = {}
    all_region_xml = region_root_el.iter("region")
    for el in all_region_xml:
        node_to_xml[el.attrib.get("name", "")] = el

    for i, block in enumerate(blocks):
        target_node = leaf_regions[i % len(leaf_regions)]
        target_xml = node_to_xml.get(target_node.name, region_root_el)
        _append_block_xml(target_xml, block)

    ET.indent(root, space="  ", level=0)
    xml_str = ET.tostring(root, encoding="unicode")
    return xml_str


def convert_file(input_path: Path, output_path: Path, limit: Optional[int] = None) -> int:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是 slide 列表")

    slides = data[:limit] if limit is not None else data
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for slide in slides:
        lines.append(slide_to_xml(slide))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(slides)


def main() -> int:
    ap = argparse.ArgumentParser(description="将 slide JSON 转为结构 XML（不含绝对坐标）")
    ap.add_argument("--input", default="data/test_with_ids.json", help="输入 JSON 路径")
    ap.add_argument("--output", default="data/test_with_ids.structure.xml", help="输出 XML 路径")
    ap.add_argument("--limit", type=int, default=None, help="仅转换前 N 条（调试用）")
    args = ap.parse_args()

    n = convert_file(Path(args.input), Path(args.output), limit=args.limit)
    print(f"[ok] converted {n} slides -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

