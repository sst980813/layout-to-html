import argparse
import html
import json
import os
import re
import sys
import textwrap
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    # Optional dependency; only used when local LLM is enabled.
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    import torch  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore


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


def _get_local_model_path_by_env() -> Optional[str]:
    return (
        os.getenv("LOCAL_MODEL_PATH")
        or os.getenv("HF_MODEL_PATH")
        or os.getenv("HUGGINGFACE_MODEL_PATH")
    )

def _extract_html_document(text: str) -> str:
    """尽量从模型输出中抽取完整 HTML 文档。"""
    if not text:
        return ""
    # 兼容大小写与空格差异
    m = re.search(r"<!doctype\s+html[^>]*>", text, flags=re.IGNORECASE)
    if m:
        return text[m.start() :].strip()
    # fallback：找到 <html
    idx = text.lower().find("<html")
    if idx != -1:
        return text[idx:].strip()
    return text.strip().replace("```html", "").replace("```", "")


@lru_cache(maxsize=1)
def _load_mlx_model_cached(model_path: str) -> Tuple[Any, Any]:
    """
    使用 mlx_lm 加载 (model, tokenizer)，并用 lru_cache 避免重复加载。
    """
    try:
        from mlx_lm.utils import load as mlx_load  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("未安装或无法加载 mlx_lm。请先安装 mlx 和 mlx_lm。") from e

    mp = (model_path or "").strip()
    if not mp:
        raise ValueError("model_path 为空，无法加载 MLX 模型。")

    # 关键修复：如果用户传的是 ./xxx 这种相对路径，
    # 但脚本当前工作目录不同步，会导致 mlx_lm 把它当成 HF repo id 去下载。
    # 这里尽量将本地存在的路径解析成绝对路径，确保走本地加载。
    p = Path(mp).expanduser()
    if not p.exists() and mp.startswith("./"):
        rel = mp[2:]
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        candidates = [
            Path(rel).expanduser(),
            (script_dir / rel).expanduser(),
            (repo_root / rel).expanduser(),
        ]
        for c in candidates:
            if c.exists():
                p = c
                break
    if p.exists():
        mp = str(p.resolve())

    model, tokenizer = mlx_load(mp, lazy=False)
    return model, tokenizer


@lru_cache(maxsize=2)
def _load_local_tokenizer_model_cached(model_path: str, resolved_device_str: str) -> Tuple[Any, Any]:
    """按 (model_path, device) 缓存 tokenizer/model，避免多 slide 时重复加载。"""
    if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
        raise RuntimeError("未安装 transformers/torch。请先 `pip install transformers torch`。")
    tokenizer_ = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # dtype 选择：
    # - cuda：fp16（速度更快）
    # - mps：fp16（主要为了缓解 OOM；采样阶段已做数值回退）
    # - cpu：float32（更保守）
    kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
    if resolved_device_str == "cuda":
        kwargs["dtype"] = torch.float16  # type: ignore[union-attr]
    elif resolved_device_str == "mps":
        kwargs["dtype"] = torch.float16  # type: ignore[union-attr]
    else:
        kwargs["dtype"] = torch.float32  # type: ignore[union-attr]

    used_device_map = False
    try:
        # 尽量避免把整个模型一次性塞进 MPS：让 HF/accelerate 自动分配到 CPU/MPS
        if resolved_device_str in ("mps", "cpu"):
            kwargs2 = dict(kwargs)
            kwargs2["device_map"] = "auto"
            used_device_map = True
            model_ = AutoModelForCausalLM.from_pretrained(model_path, **kwargs2)
        else:
            model_ = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except RuntimeError as e:
        # 如果 mps 仍然 OOM / 或者出现 invalid buffer size，退回到 CPU-only（确保至少能跑通）
        msg = str(e).lower()
        if resolved_device_str == "mps" and (
            "out of memory" in msg
            or "oom" in msg
            or "invalid buffer size" in msg
            or ("buffer size" in msg and "invalid" in msg)
        ):
            kwargs2 = dict(kwargs)
            kwargs2["dtype"] = torch.float32  # type: ignore[union-attr]
            kwargs2["device_map"] = {"": "cpu"}
            model_ = AutoModelForCausalLM.from_pretrained(model_path, **kwargs2)
            used_device_map = True
        else:
            raise e
    except TypeError as e:
        # 兼容旧版 transformers：device_map 或 dtype 参数不兼容时回退
        if "dtype" in kwargs:
            kwargs2 = dict(kwargs)
            dt = kwargs2.pop("dtype")
            kwargs2["torch_dtype"] = dt
            if resolved_device_str in ("mps", "cpu"):
                kwargs2["device_map"] = "auto"
                used_device_map = True
            model_ = AutoModelForCausalLM.from_pretrained(model_path, **kwargs2)
        else:
            # device_map 可能不支持
            model_ = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            used_device_map = False

    if not used_device_map:
        model_.to(resolved_device_str)
    model_.eval()
    return tokenizer_, model_


def _local_rewrite_html(
    blocks: List[Dict[str, Any]],
    *,
    model_path: str,
    device: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stream_stdout: bool,
) -> Optional[str]:
    """
    可选：用本地 HuggingFace LLM 在“只使用 blocks 内容”的前提下生成 HTML。
    返回完整 HTML 字符串；若不可用则返回 None。
    """
    model_path = (model_path or "").strip()
    if not model_path:
        return None

    system = (
        "你是专业演示文稿/网页排版设计师与前端工程师。"
        "你将把输入的 blocks 转换为一页现代幻灯片 HTML。"
        "硬性约束：幻灯片画布尺寸固定为 1920px * 1080px，并在 HTML/CSS 中体现（例如容器宽高）。"
        "硬性约束：禁止使用 position:absolute / position:fixed 进行布局定位（装饰性的伪元素允许使用 absolute，但不得用于排版内容）。"
        "必须优先使用 Flexbox（1D）与 CSS Grid（2D）实现布局，通过容器嵌套实现混合版式；优先使用 %、fr、clamp() 等相对单位。"
        "硬性约束：必须保留并可追踪输入数据的元素级 id 映射（见用户指令的“ID 映射规则”）。"
        "输出必须是一个完整可直接保存为 .html 的文档源码；不要输出 Markdown；不要解释。"
    )
    user = textwrap.dedent(
        f"""
        按以下“4 步流程”生成最终 HTML（最终输出只包含 HTML 源码本身）：

        第一步：感知布局结构（必须输出到 HTML 文件头部注释里）
        - 你需要对 blocks 内所有元素坐标进行空间关系分析：上下/左右关系、同一行/同一列、整体布局模式（单列/多列/网格/混合）、对齐方式、间距规律、视觉层次主次。
        - 把分析结果写成 HTML 注释，放在 <head> 里，格式为：
          <!--
          Step1 布局结构分析:
          - Layout pattern: ...
          - Rows/Cols grouping: ...
          - Alignment: ...
          - Spacing: ...
          - Visual hierarchy: ...
          -->
        - 这里是“可解释性摘要”，不要输出冗长推理过程。

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

        输入数据（只能使用这些数据的文本与 image_path）：
        {json.dumps(blocks, ensure_ascii=False)}

        请直接输出最终 HTML 文档源码。
        """
    ).strip()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        # --- MLX 推理 ---
        from mlx_lm.generate import generate as mlx_generate  # type: ignore
        from mlx_lm.generate import stream_generate as mlx_stream_generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore

        model, tokenizer = _load_mlx_model_cached(model_path)

        # 走 chat template（如果 tokenizer 支持）
        if getattr(tokenizer, "has_chat_template", False):
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=False,
                add_generation_prompt=True,
            )
            # 对齐 mlx_lm.generate CLI：chat template 分支后用 add_special_tokens=False
            prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        else:
            prompt_str = system + "\n\n" + user
            prompt_tokens = tokenizer.encode(prompt_str)

        # 避免超长：截断到 (context - max_new_tokens)
        model_max_len = getattr(tokenizer, "model_max_length", None)
        if (
            isinstance(model_max_len, int)
            and 0 < model_max_len < 100000
            and isinstance(prompt_tokens, list)
        ):
            max_prompt = model_max_len - max_new_tokens - 1
            if max_prompt > 0 and len(prompt_tokens) > max_prompt:
                prompt_tokens = prompt_tokens[-max_prompt:]

        # temperature=0 => argmax；其他情况使用采样
        sampler = make_sampler(
            temp=float(temperature),
            top_p=float(top_p),
            min_p=0.0,
            min_tokens_to_keep=1,
            top_k=0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
        )

        if stream_stdout:
            # 流式输出：边生成边打印，并累积完整文本用于抽取 HTML。
            response_text_parts: List[str] = []
            for resp in mlx_stream_generate(
                model,
                tokenizer,
                prompt_tokens,
                max_tokens=max_new_tokens,
                sampler=sampler,
            ):
                seg = getattr(resp, "text", "")
                if seg:
                    response_text_parts.append(seg)
                    sys.stdout.write(seg)
                    sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.flush()
            response_text = "".join(response_text_parts)
        else:
            response_text = mlx_generate(
                model,
                tokenizer,
                prompt_tokens,
                max_tokens=max_new_tokens,
                verbose=False,
                sampler=sampler,
            )

        return _extract_html_document(response_text)
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[MLX] 生成失败：{e}\n")
        return None


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
    ap = argparse.ArgumentParser(description="从 JSON 的 blocks 生成幻灯片 HTML（使用本地 MLX LLM 可选美化）")
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
        help="强制启用本地模型美化（需要提供 --local-model-path 或环境变量）",
    )
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="强制禁用本地模型美化",
    )
    ap.add_argument(
        "--local-model-path",
        default=_get_local_model_path_by_env() or "",
        help="本地 HuggingFace 模型目录（例如 /path/to/model）。也可通过 LOCAL_MODEL_PATH/HF_MODEL_PATH 环境变量提供。",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="推理设备：cuda|mps|cpu（默认自动选择）",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="生成最大新 token 数",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="采样温度（>0 启用采样）",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="top-p 采样参数",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="启用 MLX 流式输出：生成过程中实时打印模型文本到控制台",
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

    local_model_path = (args.local_model_path or "").strip()
    if args.no_llm:
        llm_enabled = False
    elif args.llm:
        llm_enabled = True
        if not local_model_path:
            raise ValueError("已指定 --llm，但未提供 --local-model-path（或环境变量）。")
    else:
        llm_enabled = bool(local_model_path)

    # base_dir = out_dir / "base"
    llm_dir = out_dir / "llm"
    # base_dir.mkdir(parents=True, exist_ok=True)
    if llm_enabled:
        llm_dir.mkdir(parents=True, exist_ok=True)
    for i in indices:
        slide = slides[i]
        slide_id = str(slide.get("slide_id") or "").strip()
        name = _sanitize_filename(slide_id) if slide_id else f"{prefix}_{i:04d}"
        blocks = _blocks_from_slide(slide)
        # base_html = blocks_to_base_html(blocks)
        # (base_dir / f"{name}.html").write_text(base_html, encoding="utf-8")

        if llm_enabled:
            llm_html = _local_rewrite_html(
                blocks,
                model_path=local_model_path,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                stream_stdout=bool(args.stream),
            )
            if llm_html:
                # 防御性：有时目录在循环中被意外清理/未成功创建，写文件前再确保一次。
                llm_dir.mkdir(parents=True, exist_ok=True)
                (llm_dir / f"{name}.html").write_text(llm_html, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
