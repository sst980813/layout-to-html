import argparse
import html as html_lib
from functools import lru_cache
import json
import os
import re
import time
import sys
import statistics
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from playwright.sync_api import Error as PlaywrightError  # type: ignore
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:  # pragma: no cover
    sync_playwright = None  # type: ignore
    PlaywrightError = Exception  # type: ignore

try:
    import jieba  # type: ignore
except Exception:  # pragma: no cover
    jieba = None  # type: ignore

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
    idx = text.lower().find("<html")
    if idx != -1:
        return text[idx:].strip()
    return text.strip().replace("```html", "").replace("```", "")


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h


_JS_EXTRACT_RECTS = r"""
(() => {
  const slide = document.querySelector('.slide') || document.body;
  const slideRect = slide.getBoundingClientRect();

  // Prefer data-json-id; fallback to data-block-id (兼容不同实现)
  const nodes = Array.from(document.querySelectorAll('[data-json-id], [data-block-id]'));
  const out = {};

  function visible(el) {
    const cs = window.getComputedStyle(el);
    if (!cs) return false;
    if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') === 0) return false;
    const r = el.getBoundingClientRect();
    if (!r || r.width <= 0 || r.height <= 0) return false;
    return true;
  }

  const textById = {};
  for (const el of nodes) {
    const id = (el.getAttribute('data-json-id') ?? el.getAttribute('data-block-id') ?? '').toString();
    if (!id) continue;
    if (!visible(el)) continue;
    const r = el.getBoundingClientRect();
    out[id] = {
      x: r.left - slideRect.left,
      y: r.top - slideRect.top,
      w: r.width,
      h: r.height,
    };
    textById[id] = (el.innerText || el.textContent || '').trim();
  }

  const body = document.body;
  const bodyRect = body ? body.getBoundingClientRect() : null;
  const bodyVisible = !!(bodyRect && bodyRect.width > 0 && bodyRect.height > 0);
  const hasMain = !!document.querySelector('.slide') || bodyVisible;
  const visibleText = (document.body && document.body.innerText) ? document.body.innerText.trim() : '';
  const hasAnyContent = nodes.length > 0 || visibleText.length > 0 || !!document.querySelector('img');

  return {
    rectsById: out,
    textById,
    hasMain,
    hasAnyContent,
    visibleText,
  };
})()
""".strip()


def _iter_eval_targets_from_slide(slide: Dict[str, Any]) -> Iterable[Tuple[str, Rect]]:
    """
    按“输入元素”定义生成评测目标：
    - 若 block 有 children：以每个 child 为目标（要求 child.id 存在）
    - 否则：以 block 自身为目标（使用 block.id），仅在它有可见文本或图片时作为目标
    """

    blocks = slide.get("blocks") or []
    for b in blocks:
        children = b.get("children") or []
        if children:
            for ch in children:
                json_id = "" if ch.get("id") is None else str(ch.get("id"))
                r = Rect(
                    float(ch.get("x", 0) or 0),
                    float(ch.get("y", 0) or 0),
                    float(ch.get("width", 0) or 0),
                    float(ch.get("height", 0) or 0),
                )
                yield json_id, r
        else:
            has_text = bool((b.get("text_content") or "").strip())
            has_img = bool((b.get("image_path") or "").strip())
            if not (has_text or has_img):
                continue
            json_id = "" if b.get("id") is None else str(b.get("id"))
            r = Rect(
                float(b.get("x", 0) or 0),
                float(b.get("y", 0) or 0),
                float(b.get("width", 0) or 0),
                float(b.get("height", 0) or 0),
            )
            yield json_id, r


def _iou(a: Rect, b: Rect) -> float:
    if a.w <= 0 or a.h <= 0 or b.w <= 0 or b.h <= 0:
        return 0.0
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _ref_text_elements_from_slide(slide: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    返回 (json_id, ref_text) 列表，用于按 id 匹配 HTML 中同 id 元素的文本算 ROUGE-L。
    - 有 children：kind=="text" 的 (ch.id, content)
    - 无 children：(block.id, text_content)
    """
    out: List[Tuple[str, str]] = []
    for b in slide.get("blocks") or []:
        children = b.get("children") or []
        if children:
            for ch in children:
                if str(ch.get("kind") or "").lower() == "text":
                    t = str(ch.get("content") or "")
                    if t.strip():
                        json_id = "" if ch.get("id") is None else str(ch.get("id"))
                        out.append((json_id, t))
        else:
            t = str(b.get("text_content") or "")
            if t.strip():
                json_id = "" if b.get("id") is None else str(b.get("id"))
                out.append((json_id, t))
    return out


def _tokenize_zh(s: str) -> List[str]:
    if not s or not s.strip():
        return []
    if jieba is None:
        # fallback: 按字符切分（粗糙但能跑通）
        return list(s.strip())
    return list(jieba.cut(s.strip()))


def _lcs_len_tokens(a_tokens: List[str], b_tokens: List[str]) -> int:
    """
    LCS 长度（token 序列），O(min(n,m)) 空间。
    """
    if not a_tokens or not b_tokens:
        return 0
    if len(a_tokens) < len(b_tokens):
        short, long = a_tokens, b_tokens
    else:
        short, long = b_tokens, a_tokens
    prev = [0] * (len(short) + 1)
    for tok in long:
        cur = [0]
        for j, sj in enumerate(short, start=1):
            if tok == sj:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _rouge_l_f1(candidate_tokens: List[str], reference_tokens: List[str]) -> float:
    """
    ROUGE-L F1：candidate 为生成文本 token 序列，reference 为参考 token 序列。
    Recall = LCS/len(reference), Precision = LCS/len(candidate), F1 = 2*P*R/(P+R)。
    空参考或空候选时返回 0。
    """
    if not reference_tokens:
        return 0.0
    lcs = _lcs_len_tokens(candidate_tokens, reference_tokens)
    if lcs == 0:
        return 0.0
    rec = lcs / len(reference_tokens)
    prec = lcs / len(candidate_tokens) if candidate_tokens else 0.0
    if rec + prec <= 0:
        return 0.0
    return 2.0 * rec * prec / (rec + prec)


_ID_CLEAN_RE = re.compile(r"[^a-zA-Z0-9\-_:.]+")


def _render_prompt_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    """
    默认 prompt：与现有 `generate_dpo_data.py` 的语义保持一致，
    但在本脚本中仅用于构造 DPO 样本的条件输入。
    """

    # 与 generate_dpo_data.py 的 system/user 结构保持一致：不要大改，否则训练条件会漂移。
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
    return system + "\n\n" + user


class _TagClosureHeuristicScorer:
    """
    用“轻量正则 + 栈”的方式估算 tag 是否闭合。
    注意：这是启发式，不保证完全等价于真实 HTML 规范校验。
    """

    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
    _TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9:_-]*)\b([^>]*)>", re.MULTILINE)

    def score(self, html_text: str) -> Tuple[float, Dict[str, int]]:
        if not html_text:
            return 0.0, {"unclosed": 0, "mismatch": 0, "total_start": 0}

        stack: List[str] = []
        unclosed = 0
        mismatch = 0
        total_start = 0

        # 先粗略剔除注释和 doctype，避免干扰匹配
        cleaned = re.sub(r"<!--[\s\S]*?-->", "", html_text)
        cleaned = re.sub(r"<!doctype[^>]*>", "", cleaned, flags=re.IGNORECASE)

        for m in self._TAG_RE.finditer(cleaned):
            is_close = m.group(1) == "/"
            tag = (m.group(2) or "").lower()
            attrs = m.group(3) or ""

            if not tag:
                continue
            # self-closing 形态：.../>
            is_self_closing = attrs.strip().endswith("/") or "/>" in m.group(0)

            if is_close:
                if not stack:
                    mismatch += 1
                    continue
                top = stack[-1]
                if top == tag:
                    stack.pop()
                else:
                    # 简化处理：遇到不匹配，视为一次 mismatch，并尽力恢复（弹出直到匹配或空）
                    mismatch += 1
                    while stack and stack[-1] != tag:
                        stack.pop()
                    if stack and stack[-1] == tag:
                        stack.pop()
            else:
                if tag in self._VOID_TAGS or is_self_closing:
                    continue
                total_start += 1
                stack.append(tag)

        unclosed = len(stack)
        if total_start <= 0:
            # 没有 start tag（或全部是 void/selfclosing），认为结构相对稳定
            return 1.0 if unclosed == 0 and mismatch == 0 else 0.2, {"unclosed": unclosed, "mismatch": mismatch, "total_start": total_start}

        # 归一化：错误越多分数越低
        errors = unclosed + mismatch
        closure = 1.0 - min(1.0, errors / max(1, total_start))
        return float(max(0.0, min(1.0, closure))), {"unclosed": unclosed, "mismatch": mismatch, "total_start": total_start}


def _compute_html_execution_metrics(
    *,
    page: Any,
    slide: Dict[str, Any],
    html_text: str,
    timeout_ms: int,
    tmp_html_path: Path,
    page_errors: List[str],
) -> Tuple[bool, float, float, float, Dict[str, Any]]:
    """
    使用 Playwright 渲染并抽取指标：
    返回 (valid, eiou_mean, dom_coverage, cr_mean, debug)
    """
    # 写临时 html 文件供 page.goto
    tmp_html_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_html_path.write_text(html_text, encoding="utf-8", errors="replace")

    if not tmp_html_path.exists():
        return False, 0.0, 0.0, 0.0, {"error": "missing_tmp_html"}

    valid = True
    eiou_mean = 0.0
    dom_coverage = 0.0
    cr_mean = 0.0
    debug: Dict[str, Any] = {}

    page_errors.clear()
    try:
        page.goto(tmp_html_path.as_uri(), wait_until="load", timeout=timeout_ms)
        extract: Dict[str, Any] = page.evaluate(_JS_EXTRACT_RECTS)
        has_main = bool(extract.get("hasMain"))
        has_any = bool(extract.get("hasAnyContent"))
        if page_errors:
            valid = False
            debug["page_error"] = page_errors[0]
        elif not (has_main and has_any):
            valid = False
            debug["visibility_failed"] = True
        else:
            rects_by_id = extract.get("rectsById") or {}
            text_by_id = extract.get("textById") or {}

            # layout_alignment / dom_similarity
            slide_eious: List[float] = []
            matched = 0
            total = 0
            for json_id, tgt in _iter_eval_targets_from_slide(slide):
                total += 1
                if json_id not in rects_by_id:
                    continue
                pr = rects_by_id.get(json_id)
                if not pr:
                    continue
                pred = Rect(float(pr["x"]), float(pr["y"]), float(pr["w"]), float(pr["h"]))
                slide_eious.append(_iou(tgt, pred))
                matched += 1
            dom_coverage = float(matched / total) if total > 0 else 0.0
            eiou_mean = float(statistics.mean(slide_eious)) if slide_eious else 0.0

            # structure_validity（用文本保留 CR 做代理；你也可以替换为更“结构化”的 DOM 树相似度）
            ref_elements = _ref_text_elements_from_slide(slide)
            element_f1s: List[float] = []
            for json_id, ref_elt in ref_elements:
                gen_elt = str(text_by_id.get(json_id) or "").strip()
                ref_tokens = _tokenize_zh(ref_elt)
                gen_tokens = _tokenize_zh(gen_elt)
                element_f1s.append(_rouge_l_f1(gen_tokens, ref_tokens))
            cr_mean = float(statistics.mean(element_f1s)) if element_f1s else 0.0

            debug.update(
                {
                    "matched": matched,
                    "total": total,
                    "has_main": has_main,
                    "has_any": has_any,
                    "eiou_samples": len(slide_eious),
                    "cr_elements": len(element_f1s),
                }
            )
    except PlaywrightError as e:
        valid = False
        debug["playwright_error"] = str(e)
    except Exception as e:  # pragma: no cover
        valid = False
        debug["error"] = str(e)

    return valid, eiou_mean, dom_coverage, cr_mean, debug


def score_html(
    *,
    page: Any,
    slide: Dict[str, Any],
    html_text: str,
    timeout_ms: int,
    tmp_html_path: Path,
    page_errors: List[str],
    w_layout: float = 0.4,
    w_dom: float = 0.3,
    w_struct: float = 0.2,
    w_closure: float = 0.1,
    closure_scorer: Optional[_TagClosureHeuristicScorer] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Execution-Guided Feedback Score：
      score = 0.4 * layout_alignment
            + 0.3 * dom_similarity
            + 0.2 * structure_validity
            + 0.1 * tag_closure
    """
    closure_scorer = closure_scorer or _TagClosureHeuristicScorer()
    tag_closure, closure_debug = closure_scorer.score(html_text)

    valid, eiou_mean, dom_coverage, cr_mean, exec_debug = _compute_html_execution_metrics(
        page=page,
        slide=slide,
        html_text=html_text,
        timeout_ms=timeout_ms,
        tmp_html_path=tmp_html_path,
        page_errors=page_errors,
    )

    layout_alignment = float(eiou_mean)  # 默认区间 0..1
    dom_similarity = float(dom_coverage)  # 匹配覆盖率 0..1
    structure_validity = float(cr_mean)  # 以 CR（文本保留）代理 0..1

    score = (
        w_layout * layout_alignment
        + w_dom * dom_similarity
        + w_struct * structure_validity
        + w_closure * tag_closure
    )

    metrics = {
        "valid_render": bool(valid),
        "layout_alignment": layout_alignment,
        "dom_similarity": dom_similarity,
        "structure_validity": structure_validity,
        "tag_closure": float(tag_closure),
        "closure_debug": closure_debug,
        "exec_debug": exec_debug,
        # 训练奖励可选（对 invalid 做惩罚）
        "reward": float(
            0.5 * layout_alignment
            + 0.3 * structure_validity
            + 0.2 * ((0.0 if valid else -1.0))
        ),
    }
    return float(score), metrics


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


def _extract_html_candidates_from_dir(candidate_dir: Path, slide_id: str, n: int) -> List[str]:
    """
    从目录读取候选 HTML：
    默认查找：{slide_id}*.html，按文件名排序后取前 n 个。
    """
    if not candidate_dir.exists():
        raise FileNotFoundError(f"候选目录不存在：{candidate_dir}")
    glob_pattern = f"{slide_id}*.html"
    candidates: List[str] = []
    for p in sorted(candidate_dir.glob(glob_pattern)):
        if len(candidates) >= n:
            break
        if p.is_file():
            candidates.append(p.read_text(encoding="utf-8", errors="replace"))
    return candidates


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

    # 兼容 ./xxx 这种相对路径：避免被当成 HF repo id 下载
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

    print(f"[mlx] loading model: {mp}", flush=True)
    model, tokenizer = mlx_load(mp, lazy=False)
    print("[mlx] model loaded", flush=True)
    return model, tokenizer


def _mlx_generate_html_once(
    *,
    prompt_str: str,
    model_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stream_to_stdout: bool,
) -> str:
    """
    用本地 MLX 模型对 prompt_str 生成候选 HTML。
    """
    try:
        from mlx_lm.generate import generate as mlx_generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("未安装 mlx_lm 相关依赖。请先安装 mlx 和 mlx_lm。") from e

    model, tokenizer = _load_mlx_model_cached(model_path)

    prompt_tokens = tokenizer.encode(prompt_str)
    print(
        f"[mlx] prompt_tokens={len(prompt_tokens) if isinstance(prompt_tokens, list) else 'na'} "
        f"max_new_tokens={max_new_tokens} stream={stream_to_stdout}",
        flush=True,
    )

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

    sampler = make_sampler(
        temp=float(temperature),
        top_p=float(top_p),
        min_p=0.0,
        min_tokens_to_keep=1,
        top_k=0,
        xtc_probability=0.0,
        xtc_threshold=0.0,
    )

    if stream_to_stdout:
        try:
            from mlx_lm.generate import stream_generate as mlx_stream_generate  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("启用 --stream-gen 但未找到 mlx_lm 的 stream_generate。") from e

        response_text_parts: List[str] = []
        sys.stdout.write("[mlx] streaming start\n")
        sys.stdout.flush()
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
        sys.stdout.write("\n[mlx] streaming end\n")
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

    if not response_text:
        return ""
    return _extract_html_document(str(response_text))


def generate_candidates(
    *,
    blocks: List[Dict[str, Any]],
    slide_id: str,
    n_candidates: int,
    local_model_path: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stream_gen: bool,
) -> List[str]:
    """
    Candidate Generation：
    - N 个候选：通过多次随机采样生成（使用本地 mlx_lm）
    """
    prompt = _render_prompt_from_blocks(blocks)
    model_path = (local_model_path or "").strip() or (_get_local_model_path_by_env() or "")
    if not model_path:
        raise RuntimeError("未提供本地模型路径。请设置 --local-model-path 或环境变量 LOCAL_MODEL_PATH/HF_MODEL_PATH。")
    out: List[str] = []
    for i in range(n_candidates):
        t0 = time.time()
        print(
            f"[gen] slide={slide_id} cand={i+1}/{n_candidates} temp={temperature} top_p={top_p} max_new_tokens={max_new_tokens}",
            flush=True,
        )
        html_text = _mlx_generate_html_once(
            prompt_str=prompt,
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stream_to_stdout=stream_gen,
        )
        dt = time.time() - t0
        out.append(html_text)
        print(
            f"[gen] done slide={slide_id} cand={i+1}/{n_candidates} len={len(html_text)} sec={dt:.2f}",
            flush=True,
        )
    return out


def _select_chosen_rejected(
    candidates: List[Dict[str, Any]],
    epsilon: float = 1e-9,
    rejected_strategy: str = "worst",
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Execution-Guided Preference Construction（Weak DPO）：
    chosen = sorted_candidates[0]
    rejected = sorted_candidates[-1]
    """
    if not candidates or len(candidates) < 2:
        return None

    sorted_cands = sorted(candidates, key=lambda x: float(x["score"]), reverse=True)
    chosen = sorted_cands[0]
    if rejected_strategy == "worst":
        rejected = sorted_cands[-1]
    elif rejected_strategy == "random-worst":
        # 仍然是“弱偏好”，但避免 extreme tie：在后 30% 范围内随机选 worst
        import random

        cut = max(1, int(len(sorted_cands) * 0.3))
        rejected = random.choice(sorted_cands[-cut:])
    else:
        raise ValueError(f"未知 rejected_strategy：{rejected_strategy}")

    if abs(float(chosen["score"]) - float(rejected["score"])) <= epsilon:
        # 分数几乎相同，不形成有效偏好对
        return None
    return chosen, rejected


def main() -> int:
    ap = argparse.ArgumentParser(description="Execution-Guided Preference Optimization（Weak DPO 数据集生成）")
    ap.add_argument("--input", default=str(Path("data/test_with_ids.json")), help="输入 blocks JSON（默认 data/test_with_ids.json）")
    ap.add_argument("--output", default=str(Path("data/egpo_weak_dpo.jsonl")), help="输出 JSONL 路径")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")

    ap.add_argument("--num-candidates", type=int, default=4, help="每个 slide 生成/加载的候选数量 N（默认 4）")
    ap.add_argument("--timeout-ms", type=int, default=80000, help="Playwright 单页超时（默认 8000ms）")

    ap.add_argument("--candidate-dir", default="", help="候选 HTML 目录（可选）。若提供，会从目录读取 {slide_id}*.html 作为候选。")
    ap.add_argument("--candidate-dir-limit", type=int, default=None, help="候选目录模式下，每 slide 最多取前多少个候选（默认使用 --num-candidates）")
    ap.add_argument("--local-model-path", default=_get_local_model_path_by_env() or "", help="本地 MLX 模型路径（默认从 LOCAL_MODEL_PATH/HF_MODEL_PATH/HUGGINGFACE_MODEL_PATH 读取）")
    ap.add_argument("--temperature", type=float, default=0.7, help="采样温度（默认 0.7）")
    ap.add_argument("--top-p", type=float, default=0.9, help="top-p（默认 0.9）")
    ap.add_argument("--max-new-tokens", type=int, default=4096, help="最大生成 token（默认 700）")

    ap.add_argument("--w-layout", type=float, default=0.4, help="score 分量：layout_alignment 权重（默认 0.4）")
    ap.add_argument("--w-dom", type=float, default=0.3, help="score 分量：dom_similarity 权重（默认 0.3）")
    ap.add_argument("--w-struct", type=float, default=0.2, help="score 分量：structure_validity 权重（默认 0.2）")
    ap.add_argument("--w-closure", type=float, default=0.1, help="score 分量：tag_closure 权重（默认 0.1）")

    ap.add_argument("--epsilon", type=float, default=1e-9, help="chosen 与 rejected 分数差小于该值时跳过（默认 1e-9）")
    ap.add_argument("--rejected-strategy", default="worst", choices=["worst", "random-worst"], help="rejected 选择策略（默认 worst）")
    ap.add_argument("--stream-gen", action="store_true", help="生成候选时流式输出模型文本到控制台")

    args = ap.parse_args()

    print(
        "[start] EGPO Weak DPO dataset generation\n"
        f"  input={args.input}\n"
        f"  output={args.output}\n"
        f"  limit={args.limit}\n"
        f"  num_candidates={args.num_candidates}\n"
        f"  timeout_ms={args.timeout_ms}\n"
        f"  mlx_model_path={args.local_model_path or _get_local_model_path_by_env() or ''}\n"
        f"  sampling: temperature={args.temperature} top_p={args.top_p} max_new_tokens={args.max_new_tokens}\n"
        f"  score_weights: w_layout={args.w_layout} w_dom={args.w_dom} w_struct={args.w_struct} w_closure={args.w_closure}\n",
        flush=True,
    )

    if sync_playwright is None:
        raise RuntimeError(
            "未能导入 playwright（sync_playwright=None）。"
            f"\n当前 Python：{sys.executable}"
            f"\n当前版本：{sys.version}"
            "\n请确认已在“同一个 python 环境”里执行："
            "`pip install playwright` 并执行 `playwright install chromium`。"
        )

    slides = _load_slides_from_json(Path(args.input))
    if args.limit is not None:
        slides = slides[: max(0, args.limit)]

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped_no_candidates = 0
    skipped_no_pair = 0
    total_slides = len(slides)
    print(f"[start] loaded slides: {total_slides}", flush=True)

    closure_scorer = _TagClosureHeuristicScorer()

    # 单独开 Playwright 用于渲染打分（尽可能复用 browser/page）
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page = context.new_page()

        # 用于捕获 pageerror（兼容各种渲染异常）
        page_errors: List[str] = []

        def _on_page_error(e: Any) -> None:
            try:
                page_errors.append(str(e))
            except Exception:
                page_errors.append("pageerror")

        page.on("pageerror", _on_page_error)

        # 必须用绝对路径：Playwright file URI 不能使用相对路径
        tmp_html_path = (Path(".tmp_egpo_html") / "candidate.html").resolve()

        with out_path.open("w", encoding="utf-8") as f:
            for idx, slide in enumerate(slides):
                slide_id = str(slide.get("slide_id") or "").strip()
                if not slide_id:
                    continue

                print(f"[slide] ({idx+1}/{total_slides}) slide_id={slide_id} start", flush=True)

                blocks = _blocks_from_slide(slide)
                prompt = _render_prompt_from_blocks(blocks)

                candidates_html: List[str] = []
                if args.candidate_dir:
                    cand_dir = Path(args.candidate_dir).resolve()
                    limit_n = args.candidate_dir_limit or args.num_candidates
                    candidates_html = _extract_html_candidates_from_dir(cand_dir, slide_id=slide_id, n=limit_n)
                else:
                    candidates_html = generate_candidates(
                        blocks=blocks,
                        slide_id=slide_id,
                        n_candidates=args.num_candidates,
                        local_model_path=args.local_model_path,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        stream_gen=bool(args.stream_gen),
                    )

                print(
                    f"[slide] slide_id={slide_id} got candidates={len(candidates_html)}",
                    flush=True,
                )

                # 需要至少 2 个候选才能构造 chosen/rejected
                if len(candidates_html) < 2:
                    skipped_no_candidates += 1
                    print(f"[slide] slide_id={slide_id} skip (<2 candidates)", flush=True)
                    continue

                scored: List[Dict[str, Any]] = []
                print(
                    f"[score] slide_id={slide_id} start scoring {len(candidates_html)} candidates",
                    flush=True,
                )
                for j, cand_html in enumerate(candidates_html):
                    print(
                        f"[score] slide_id={slide_id} cand={j+1}/{len(candidates_html)} start",
                        flush=True,
                    )
                    t0 = time.time()
                    score, metrics = score_html(
                        page=page,
                        slide=slide,
                        html_text=cand_html,
                        timeout_ms=args.timeout_ms,
                        tmp_html_path=tmp_html_path,
                        page_errors=page_errors,
                        w_layout=args.w_layout,
                        w_dom=args.w_dom,
                        w_struct=args.w_struct,
                        w_closure=args.w_closure,
                        closure_scorer=closure_scorer,
                    )
                    dt = time.time() - t0
                    scored.append(
                        {
                            "candidate_index": j,
                            "html": cand_html,
                            "score": score,
                            "metrics": metrics,
                        }
                    )
                    print(
                        f"[score] slide_id={slide_id} cand={j+1}/{len(candidates_html)} score={score:.4f} "
                        f"layout={metrics.get('layout_alignment', 0.0):.4f} dom={metrics.get('dom_similarity', 0.0):.4f} "
                        f"struct={metrics.get('structure_validity', 0.0):.4f} closure={metrics.get('tag_closure', 0.0):.4f} "
                        f"sec={dt:.2f}",
                        flush=True,
                    )

                chosen_rejected = _select_chosen_rejected(
                    candidates=scored,
                    epsilon=args.epsilon,
                    rejected_strategy=args.rejected_strategy,
                )
                if chosen_rejected is None:
                    skipped_no_pair += 1
                    print(f"[slide] slide_id={slide_id} skip (no valid chosen/rejected)", flush=True)
                    continue

                chosen, rejected = chosen_rejected
                print(
                    f"[pair] slide_id={slide_id} chosen_score={chosen['score']:.4f} rejected_score={rejected['score']:.4f}",
                    flush=True,
                )

                obj = {
                    "id": slide_id,
                    "slide_id": slide_id,
                    "prompt": prompt,
                    "chosen": chosen["html"],
                    "rejected": rejected["html"],
                    "chosen_score": chosen["score"],
                    "rejected_score": rejected["score"],
                    "chosen_metrics": chosen.get("metrics") or {},
                    "rejected_metrics": rejected.get("metrics") or {},
                    # 便于调参/复现：记录全部候选分数
                    "all_candidates": [
                        {"candidate_index": c["candidate_index"], "score": c["score"]}
                        for c in scored
                    ],
                }

                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                created += 1

        context.close()
        browser.close()

    summary = {
        "input": str(args.input),
        "output": str(out_path),
        "total_slides": len(slides),
        "created_pairs": created,
        "skipped_no_candidates": skipped_no_candidates,
        "skipped_no_pair": skipped_no_pair,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

