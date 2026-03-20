import argparse
import html
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    # Optional dependency; only used when LLM is enabled.
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # type: ignore


@dataclass(frozen=True)
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> int:
        return int(round(self.max_x - self.min_x))

    @property
    def height(self) -> int:
        return int(round(self.max_y - self.min_y))


def _iter_items_from_blocks(blocks: List[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for b in blocks:
        yield {
            "kind": "block",
            "x": b.get("x", 0),
            "y": b.get("y", 0),
            "width": b.get("width", 0),
            "height": b.get("height", 0),
            "type": b.get("type", ""),
            "text_content": b.get("text_content", ""),
        }
        for ch in (b.get("children") or []):
            yield ch


def _compute_bbox(blocks: List[Dict[str, Any]]) -> BBox:
    items = list(_iter_items_from_blocks(blocks))
    if not items:
        return BBox(0, 0, 1, 1)

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for it in items:
        x = float(it.get("x", 0) or 0)
        y = float(it.get("y", 0) or 0)
        w = float(it.get("width", 0) or 0)
        h = float(it.get("height", 0) or 0)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + max(w, 0))
        max_y = max(max_y, y + max(h, 0))

    if not (min_x < float("inf") and min_y < float("inf") and max_x > float("-inf") and max_y > float("-inf")):
        return BBox(0, 0, 1, 1)

    # Avoid zero-size canvases.
    if max_x - min_x < 1:
        max_x = min_x + 1
    if max_y - min_y < 1:
        max_y = min_y + 1

    return BBox(min_x, min_y, max_x, max_y)


def _style_abs(x: float, y: float, w: float, h: float, offset_x: float, offset_y: float) -> str:
    left = x - offset_x
    top = y - offset_y
    return f"left:{left:.0f}px;top:{top:.0f}px;width:{w:.0f}px;height:{h:.0f}px;"


def _tag_for_semantic_level(level: Optional[int]) -> str:
    # A light mapping: 0..3 -> span/h2/h1/p (common in slide extraction)
    if level is None:
        return "div"
    try:
        lv = int(level)
    except Exception:
        return "div"
    if lv <= 0:
        return "span"
    if lv == 1:
        return "h2"
    if lv == 2:
        return "h1"
    return "p"


_ID_CLEAN_RE = re.compile(r"[^a-zA-Z0-9\-_:.]+")


def _to_html_id(raw_id: Any, *, prefix: str) -> Tuple[str, str]:
    """
    返回 (html_id, data_json_id)：
    - data_json_id：输入 id 的字符串化（用于精确映射）
    - html_id：合法的 HTML id（必要时会加前缀/清洗）
    """
    data_json_id = "" if raw_id is None else str(raw_id)
    cleaned = _ID_CLEAN_RE.sub("-", data_json_id.strip())
    if not cleaned:
        cleaned = prefix
    # HTML id 建议以字母开头；若不是则加前缀。
    if not re.match(r"^[A-Za-z]", cleaned):
        cleaned = f"{prefix}-{cleaned}"
    return cleaned, data_json_id


def blocks_to_base_html(blocks: List[Dict[str, Any]]) -> str:
    """
    仅使用 blocks 内的几何与内容信息，生成一页“还原布局”的幻灯片 HTML。
    """
    # 约束：幻灯片画布固定 1920x1080
    slide_w, slide_h = 1920, 1080
    # blocks 坐标通常已在 1920x1080 画布内；不再按 bbox 裁剪/偏移
    offset_x, offset_y = 0.0, 0.0

    parts: List[str] = []
    parts.append("<!doctype html>")
    parts.append('<html lang="zh-CN">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8" />')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1" />')
    parts.append("<title>Slide</title>")
    parts.append(
        "<style>"
        ":root{--bg:#0b0f17;--fg:#e9eef8;--muted:#b8c2d6;}"
        "html,body{height:100%;margin:0;background:var(--bg);color:var(--fg);}"
        "body{display:flex;align-items:center;justify-content:center;}"
        ".stage{padding:24px;}"
        ".slide{position:relative;background:#0e1420;"
        "box-shadow:0 30px 120px rgba(0,0,0,.55);overflow:hidden;"
        f"width:{slide_w}px;height:{slide_h}px;"
        "}"
        ".el{position:absolute;box-sizing:border-box;}"
        ".text{white-space:pre-wrap;line-height:1.25;color:var(--fg);}"
        ".text span,.text p,.text h1,.text h2{margin:0;}"
        ".img{object-fit:cover;}"
        ".debug{outline:1px dashed rgba(255,255,255,.12);}"
        "</style>"
    )
    parts.append("</head>")
    parts.append("<body>")
    parts.append('<div class="stage">')
    parts.append(f'<div class="slide" data-width="{slide_w}" data-height="{slide_h}">')

    # Render blocks by preferring children when present (richer granularity).
    seen_ids: Dict[str, int] = {}
    for b in blocks:
        children = b.get("children") or []
        if children:
            for ch in children:
                kind = ch.get("kind")
                x = float(ch.get("x", 0) or 0)
                y = float(ch.get("y", 0) or 0)
                w = float(ch.get("width", 0) or 0)
                h = float(ch.get("height", 0) or 0)
                style = _style_abs(x, y, w, h, offset_x, offset_y)
                html_id, json_id = _to_html_id(ch.get("id"), prefix="el")
                # 去重：重复时加 -2/-3...（仍保留 data-json-id 原值）
                cnt = seen_ids.get(html_id, 0) + 1
                seen_ids[html_id] = cnt
                if cnt > 1:
                    html_id = f"{html_id}-{cnt}"

                if kind == "image":
                    src = html.escape(str(ch.get("image_path", "")))
                    parts.append(
                        f'<img id="{html.escape(html_id)}" data-json-id="{html.escape(json_id)}" class="el img" style="{style}" src="{src}" alt="" />'
                    )
                else:
                    content = html.escape(str(ch.get("content", "")))
                    tag = _tag_for_semantic_level(ch.get("semantic_level"))
                    font_size = ch.get("font_size")
                    fs = ""
                    if font_size is not None:
                        try:
                            fs = f"font-size:{float(font_size):.0f}px;"
                        except Exception:
                            fs = ""
                    parts.append(
                        f'<div id="{html.escape(html_id)}" data-json-id="{html.escape(json_id)}" class="el text" style="{style}{fs}"><{tag}>{content}</{tag}></div>'
                    )
        else:
            # Fallback: render the block's text_content if no children.
            if b.get("type") == "image":
                x = float(b.get("x", 0) or 0)
                y = float(b.get("y", 0) or 0)
                w = float(b.get("width", 0) or 0)
                h = float(b.get("height", 0) or 0)
                style = _style_abs(x, y, w, h, offset_x, offset_y)
                html_id, json_id = _to_html_id(b.get("id"), prefix="block")
                cnt = seen_ids.get(html_id, 0) + 1
                seen_ids[html_id] = cnt
                if cnt > 1:
                    html_id = f"{html_id}-{cnt}"
                src = html.escape(str(b.get("image_path", "")))
                parts.append(
                    f'<img id="{html.escape(html_id)}" data-json-id="{html.escape(json_id)}" class="el img" style="{style}" src="{src}" alt="" />'
                )
            else:
                bt = (b.get("text_content") or "").strip()
                if not bt:
                    continue
                x = float(b.get("x", 0) or 0)
                y = float(b.get("y", 0) or 0)
                w = float(b.get("width", 0) or 0)
                h = float(b.get("height", 0) or 0)
                style = _style_abs(x, y, w, h, offset_x, offset_y)
                html_id, json_id = _to_html_id(b.get("id"), prefix="block")
                cnt = seen_ids.get(html_id, 0) + 1
                seen_ids[html_id] = cnt
                if cnt > 1:
                    html_id = f"{html_id}-{cnt}"
                parts.append(
                    f'<div id="{html.escape(html_id)}" data-json-id="{html.escape(json_id)}" class="el text" style="{style}"><p>{html.escape(bt)}</p></div>'
                )

    parts.append("</div></div></body></html>")
    return "\n".join(parts)


def _llm_enabled_by_env() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _blocks_to_flat_elements_json_for_prompt(blocks: List[Dict[str, Any]]) -> str:
    """
    将 blocks 转为扁平元素 JSON 数组（用于提示模型理解布局）。
    规则：
    - 若 block 有 children：block 不加入，仅展开 children；
    - 若 block 无 children：输出 block 自身；
    - 保留原始字段（如 x/y/width/height/kind/content/...）。
    """
    elements: List[Dict[str, Any]] = []
    for b in blocks:
        children = b.get("children") or []
        if children:
            for ch in children:
                elements.append(dict(ch))
        else:
            block_item = {
                "id": b.get("id"),
                "kind": b.get("type"),
                "x": b.get("x"),
                "y": b.get("y"),
                "width": b.get("width"),
                "height": b.get("height"),
            }
            if b.get("text_content"):
                block_item["content"] = b.get("text_content")
            if b.get("image_path"):
                block_item["image_path"] = b.get("image_path")
            elements.append(block_item)

    return json.dumps(elements, ensure_ascii=False)


def _blocks_to_structure_xml_for_prompt(blocks: List[Dict[str, Any]]) -> str:
    """
    将 blocks 转为结构化 XML（用于提示模型理解层次关系）。
    - 保留 block/children 层次；
    - 提供 x/y/width/height 作为几何参考；
    - 仅用于提示，不作为训练标签。
    """
    lines: List[str] = []
    lines.append("<layout_reference>")
    lines.append('  <canvas width="1920" height="1080" />')
    lines.append("  <blocks>")

    for b in blocks:
        block_id = html.escape(str(b.get("id", "")))
        block_type = html.escape(str(b.get("type", "")))
        x = html.escape(str(b.get("x", "")))
        y = html.escape(str(b.get("y", "")))
        w = html.escape(str(b.get("width", "")))
        h = html.escape(str(b.get("height", "")))
        lines.append(
            f'    <block id="{block_id}" type="{block_type}" x="{x}" y="{y}" width="{w}" height="{h}">'
        )

        children = b.get("children") or []
        if children:
            lines.append("      <children>")
            for ch in children:
                cid = html.escape(str(ch.get("id", "")))
                kind = html.escape(str(ch.get("kind", "")))
                cx = html.escape(str(ch.get("x", "")))
                cy = html.escape(str(ch.get("y", "")))
                cw = html.escape(str(ch.get("width", "")))
                chh = html.escape(str(ch.get("height", "")))
                if kind == "text":
                    txt = html.escape(str(ch.get("content", "")))
                    sem = html.escape(str(ch.get("semantic_level", "")))
                    fs = html.escape(str(ch.get("font_size", "")))
                    lines.append(
                        f'        <node id="{cid}" kind="{kind}" x="{cx}" y="{cy}" width="{cw}" height="{chh}" semantic_level="{sem}" font_size="{fs}">{txt}</node>'
                    )
                elif kind == "image":
                    src = html.escape(str(ch.get("image_path", "")))
                    lines.append(
                        f'        <node id="{cid}" kind="{kind}" x="{cx}" y="{cy}" width="{cw}" height="{chh}" src="{src}" />'
                    )
                else:
                    lines.append(
                        f'        <node id="{cid}" kind="{kind}" x="{cx}" y="{cy}" width="{cw}" height="{chh}" />'
                    )
            lines.append("      </children>")
        else:
            text_content = html.escape(str(b.get("text_content", "")))
            img_path = html.escape(str(b.get("image_path", "")))
            if text_content:
                lines.append(f"      <text>{text_content}</text>")
            if img_path:
                lines.append(f'      <image src="{img_path}" />')

        lines.append("    </block>")

    lines.append("  </blocks>")
    lines.append("</layout_reference>")
    return "\n".join(lines)


def _openai_rewrite_html(blocks: List[Dict[str, Any]], *, stream_to_stdout: bool = True) -> Optional[str]:
    """
    可选：用大模型在“只使用 blocks 内容”的前提下生成 HTML。
    返回完整 HTML 字符串；若不可用则返回 None。
    """
    api_key = "sk-byd4ak5ejmws24xe"
    if not api_key:
        return None
    if openai is None:
        raise RuntimeError("检测到 LLM_API_KEY/OPENAI_API_KEY，但未安装 openai。请先 `pip install openai`。")

    model = os.getenv("LLM_MODEL", "deepseek-v3.2")
    base_url = os.getenv("LLM_BASE_URL", "https://cloud.infini-ai.com/maas/v1")
    llm_client = openai.OpenAI(api_key=api_key or "not-needed", base_url=base_url)

    system = (
        "你是专业演示文稿/网页排版设计师与前端工程师。"
        "你将把输入的 blocks 转换为一页现代幻灯片 HTML。"
        "硬性约束：幻灯片画布尺寸固定为 1920px * 1080px，并在 HTML/CSS 中体现（例如容器宽高）。"
        "硬性约束：禁止使用 position:absolute / position:fixed 进行布局定位（装饰性的伪元素允许使用 absolute，但不得用于排版内容）。"
        "必须优先使用 Flexbox（1D）与 CSS Grid（2D）实现布局，通过容器嵌套实现混合版式；优先使用 %、fr、clamp() 等相对单位。"
        "硬性约束：必须保留并可追踪输入数据的元素级 id 映射（见用户指令的“ID 映射规则”）。"
        "输出必须是一个完整可直接保存为 .html 的文档源码；不要输出 Markdown；不要解释。"
    )
    flat_elements_json = _blocks_to_flat_elements_json_for_prompt(blocks)
    structure_xml = _blocks_to_structure_xml_for_prompt(blocks)
    user = textwrap.dedent(
        f"""
        按以下“4 步流程”生成最终 HTML（最终输出只包含 HTML 源码本身）：

        第一步：基于“结构 XML + 扁平元素数组”进行布局规划（无需输出分析注释）
        - 你会同时收到结构 XML（用于理解层次关系）和扁平元素 JSON 数组（用于快速读取关键几何要素）。
        - 你会收到一个扁平元素 JSON 数组（非树形），数组中每一项对应一个输入元素。
        - 若原 block 有 children，则 block 本身不进入数组，仅将 children 的每个元素作为独立数组项。
        - 若 block 无 children，则该 block 本身进入数组。
        - 数组中包含 x/y/width/height，可作为布局参考；你需要综合这些信息决定整体布局（单列/多列/网格/混合）以及局部元素排布方式。
        - 不要在最终 HTML 中输出“步骤分析”注释，直接给出实现后的代码。

        第二步：用现代布局方式实现（禁止 absolute 硬编码）
        - 不允许对内容元素使用 position:absolute/fixed 做布局定位。
        - 用 Grid/Flex + 容器嵌套实现，与 blocks 的几何结构尽量一致（允许合理重排以提升可读性）。
        - 用相对单位（%、fr、clamp）与 max-width 约束，保证可维护性。
        - 幻灯片画布固定为 1920px*1080px（可以用一个固定尺寸的 slide 容器承载）。
        注意生成的html需要满足下面的映射规则：
         - 你会收到一组 blocks（每个 block 可能有 children）。你必须为“每一个输入元素”生成一个对应的可见 DOM 元素，并为该 DOM 元素同时写入：
          - id="<html_id>"：合法且唯一的 HTML id
          - data-json-id="<json_id>"：严格等于输入元素的 id（字符串化），用于精确映射回数据集
        - “输入元素”的定义：
          - 若某个 block 存在 children：以每个 child 作为输入元素（而不是 block 本身）
          - 若某个 block 没有 children：以该 block 作为输入元素（仅当它有可见文本或图片）
        - 不允许丢失任何输入元素；允许额外添加容器与装饰元素，但容器/装饰元素不得挪用输入元素的 data-json-id。
        - html_id 生成规则（请在输出中严格遵守）：
          - json_id = str(输入元素的 id)；若输入 id 为 null/缺失，则 json_id 设为 ""（空字符串）
        - 重要：输入元素的文本与图片必须出现在带 data-json-id 的同一个元素上（例如 <div ...>文字</div> 或 <img .../>），不要把 data-json-id 放到外层容器而内容放到内层。

        第三步：设计配色方案和视觉样式
        - 背景优先用线性/径向渐变，不要纯色单调背景。
        - 标题与背景强对比；不同层级文字用色彩深度区分权重。
        - 内容区域用半透明卡片背景增强层次。

        第四步：添加装饰元素与视觉细节
        - 用伪元素增加背景装饰形状（可用 absolute，但只用于装饰，不得用于排版）。
        - 添加彩色阴影、圆角；允许渐变文字（background-clip:text）。
        - 图片可加彩色边框与轻量滤镜（如 saturate/contrast），但不得改动图片内容。
        - 文字区域可加入图标icon增加丰富度。

        结构 XML（用于层次结构推断）：
        {structure_xml}

        扁平元素数组（用于布局推断）：
        {flat_elements_json}

        请直接输出最终 HTML 文档源码。
        """
    ).strip()

    content_parts: List[str] = []

    print(user)
    if stream_to_stdout:
        stream = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            stream=True,
        )
        for event in stream:
            delta = None
            try:
                delta = event.choices[0].delta.content  # type: ignore[attr-defined]
            except Exception:
                delta = None
            if delta:
                content_parts.append(delta)
                sys.stdout.write(delta)
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        content = "".join(content_parts)
    else:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        try:
            content = resp.choices[0].message.content  # type: ignore[attr-defined]
        except Exception:
            content = None
    if not content or not isinstance(content, str):
        return None
    return content.strip()


def _load_slides_from_json(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("输入 JSON 顶层应为非空数组")
    return raw


def _blocks_from_slide(slide: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = slide.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("slide.blocks 不是数组")
    return blocks


def _sanitize_filename(s: str) -> str:
    """将 slide_id 转为安全文件名（替换非法字符）。"""
    if not s:
        return ""
    bad = '\\/:*?"<>|'
    for c in bad:
        s = s.replace(c, "_")
    return s.strip() or "slide"


def _resolve_outputs(out_arg: str, total: int) -> Tuple[Path, str]:
    """
    返回 (输出目录, 文件名前缀)
    - out_arg 指向目录：直接使用该目录，前缀为 'slide'
    - out_arg 指向 .html 文件：使用其父目录，前缀为该文件 stem
    - out_arg 无后缀：按目录处理
    """
    p = Path(out_arg)
    if p.suffix.lower() == ".html":
        return (p.parent, p.stem)
    # Treat as directory (even if it doesn't exist yet)
    return (p, "slide")


def main() -> int:
    ap = argparse.ArgumentParser(description="从 data/test.json 的 blocks 生成幻灯片 HTML（可选 LLM 美化）")
    ap.add_argument("--input", default=str(Path("data/test_with_ids.json")), help="输入 JSON（默认 data/test_with_ids.json）")
    ap.add_argument(
        "--slide-index",
        type=int,
        default=None,
        help="只生成第几个 slide（不传则生成 input 中所有 slides）",
    )
    ap.add_argument(
        "--out",
        default=str(Path("generate_html/output/")),
        help="输出路径：生成单页可给 .html 文件；生成多页建议给目录（默认 generate_html/output/）",
    )
    ap.add_argument(
        "--llm",
        action="store_true",
        help="强制启用大模型美化（会额外产出 *.llm.html）",
    )
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="强制禁用大模型美化（忽略环境变量自动启用）",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    slides = _load_slides_from_json(in_path)

    out_dir, prefix = _resolve_outputs(args.out, len(slides))
    out_dir.mkdir(parents=True, exist_ok=True)

    indices: List[int]
    if args.slide_index is None:
        indices = list(range(len(slides)))
    else:
        if args.slide_index < 0 or args.slide_index >= len(slides):
            raise IndexError(f"slide_index 超出范围：0..{len(slides)-1}")
        indices = [args.slide_index]

    llm_enabled = True
    # rule_dir = out_dir / "rule"
    llm_dir = out_dir / "llm"
    # rule_dir.mkdir(parents=True, exist_ok=True)
    if llm_enabled:
        llm_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        slide = slides[i]
        slide_id = str(slide.get("slide_id") or "").strip()
        name = _sanitize_filename(slide_id) if slide_id else f"{prefix}_{i:04d}"
        blocks = _blocks_from_slide(slide)
        # base_html = blocks_to_base_html(blocks)
        # out_path = rule_dir / f"{name}.html"
        # out_path.write_text(base_html, encoding="utf-8")

        if llm_enabled:
            llm_html = _openai_rewrite_html(blocks, stream_to_stdout=True)
            if llm_html:
                llm_path = llm_dir / f"{name}.html"
                llm_path.write_text(llm_html, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
