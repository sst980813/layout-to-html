import argparse
import json
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


def _iter_eval_targets_from_slide(slide: Dict[str, Any]) -> Iterable[Tuple[str, Rect]]:
    """
    按“输入元素”定义生成评测目标：
    - 若 block 有 children：以每个 child 为目标（要求 child.id 存在）
    - 否则：以 block 自身为目标（使用 block.id）
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
            # 仅在确实有可见内容时作为目标（避免把纯容器块也算进来）
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
"""


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


def _resolve_html_path(html_dir: Path, slide_id: str) -> Path:
    return (html_dir / f"{slide_id}.html").resolve()


def _render_prompt_from_blocks(blocks: List[Dict[str, Any]]) -> str:
    # 与 `generate_html/generate.py` 的（重写 HTML）prompt 语义保持一致，
    # 但这里仅作为 DPO 数据的条件输入。
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


def _score_eiou_for_html(
    *,
    page: Any,
    slide: Dict[str, Any],
    html_path: Path,
    timeout_ms: int,
    page_errors: List[str],
) -> Tuple[int, Optional[float]]:
    """
    返回 (valid, eiou_mean)：
    - valid=1 表示与 `evaluate.py` 相同的渲染可见性约束通过
    - eiou_mean：按 id 匹配的元素 IoU 均值；若没有任何 matched，则返回 0.0
    """

    if not html_path.exists():
        return 0, None

    valid = 1
    try:
        page_errors.clear()
        page.goto(html_path.as_uri(), wait_until="load", timeout=timeout_ms)
        extract: Dict[str, Any] = page.evaluate(_JS_EXTRACT_RECTS)
        has_main = bool(extract.get("hasMain"))
        has_any = bool(extract.get("hasAnyContent"))
        if page_errors:
            return 0, None
        if not (has_main and has_any):
            return 0, None

        rects_by_id = extract.get("rectsById") or {}
        slide_eious: List[float] = []
        for json_id, tgt in _iter_eval_targets_from_slide(slide):
            pr = rects_by_id.get(json_id)
            if not pr:
                continue
            pred = Rect(float(pr["x"]), float(pr["y"]), float(pr["w"]), float(pr["h"]))
            slide_eious.append(_iou(tgt, pred))
        eiou_mean = float(statistics.mean(slide_eious)) if slide_eious else 0.0
        return valid, eiou_mean
    except PlaywrightError:
        return 0, None
    except Exception:  # pragma: no cover
        return 0, None


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 DPO 用户偏好对（基于 EIoU 更优 HTML 作为 chosen）")
    ap.add_argument("--input", default=str(Path("data/test_with_ids.json")), help="带 id 的布局 JSON（默认 data/test_with_ids.json）")
    ap.add_argument("--html-dir-a", required=True, help="候选 HTML 目录 A（slide_id.html）")
    ap.add_argument("--html-dir-b", required=True, help="候选 HTML 目录 B（slide_id.html）")
    ap.add_argument("--output", default=str(Path("data/dpo_preferences.jsonl")), help="输出 JSONL 路径（默认 data/dpo_preferences.jsonl）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    ap.add_argument("--timeout-ms", type=int, default=8000, help="单页加载超时（默认 8000ms）")
    ap.add_argument("--headless", action="store_true", help="强制无头（覆盖 --headed）")
    ap.add_argument("--headed", action="store_true", help="有头模式（默认无头）")
    ap.add_argument("--validity-weight", type=float, default=1.0, help="validity(0/1) 在综合打分中的权重（默认 1.0）")
    ap.add_argument("--eiou-weight", type=float, default=1.0, help="EIoU 在综合打分中的权重（默认 1.0）")
    ap.add_argument("--epsilon", type=float, default=1e-9, help="EIoU 差值小于该值则视为平局跳过（默认 1e-9）")
    args = ap.parse_args()

    if sync_playwright is None:
        raise RuntimeError("未安装 playwright。请先 `pip install playwright` 并执行 `playwright install chromium`。")

    slides: List[Dict[str, Any]] = _load_slides_from_json(Path(args.input))
    if args.limit is not None:
        slides = slides[: max(0, args.limit)]

    html_dir_a = Path(args.html_dir_a).resolve()
    html_dir_b = Path(args.html_dir_b).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headless = True
    if args.headed:
        headless = False
    if args.headless:
        headless = True

    created = 0
    skipped_missing = 0
    skipped_invalid = 0
    skipped_tie = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page = context.new_page()

        page_errors: List[str] = []

        def _on_page_error(e: Any) -> None:
            try:
                page_errors.append(str(e))
            except Exception:
                page_errors.append("pageerror")

        page.on("pageerror", _on_page_error)

        with out_path.open("w", encoding="utf-8") as f:
            for idx, slide in enumerate(slides):
                slide_id = str(slide.get("slide_id") or "").strip()
                if not slide_id:
                    skipped_missing += 1
                    continue

                html_path_a = _resolve_html_path(html_dir_a, slide_id)
                html_path_b = _resolve_html_path(html_dir_b, slide_id)
                if not html_path_a.exists() or not html_path_b.exists():
                    skipped_missing += 1
                    continue

                valid_a, eiou_a = _score_eiou_for_html(
                    page=page,
                    slide=slide,
                    html_path=html_path_a,
                    timeout_ms=args.timeout_ms,
                    page_errors=page_errors,
                )
                valid_b, eiou_b = _score_eiou_for_html(
                    page=page,
                    slide=slide,
                    html_path=html_path_b,
                    timeout_ms=args.timeout_ms,
                    page_errors=page_errors,
                )

                eiou_a_val = float(eiou_a) if eiou_a is not None else 0.0
                eiou_b_val = float(eiou_b) if eiou_b is not None else 0.0

                # 综合打分：把单条有效性(valid)当作 validity_rate(0/1) 的代理，再与 EIoU 加权融合。
                score_a = args.validity_weight * float(valid_a) + args.eiou_weight * eiou_a_val
                score_b = args.validity_weight * float(valid_b) + args.eiou_weight * eiou_b_val

                if valid_a == 0 and valid_b == 0:
                    skipped_invalid += 1
                    continue

                if abs(score_a - score_b) <= args.epsilon:
                    skipped_tie += 1
                    continue

                if score_a > score_b:
                    chosen_path, rejected_path = html_path_a, html_path_b
                    chosen_score, rejected_score = eiou_a_val, eiou_b_val
                    chosen_dir, rejected_dir = "a", "b"
                else:
                    chosen_path, rejected_path = html_path_b, html_path_a
                    chosen_score, rejected_score = eiou_b_val, eiou_a_val
                    chosen_dir, rejected_dir = "b", "a"

                blocks = _blocks_from_slide(slide)
                prompt = _render_prompt_from_blocks(blocks)

                chosen_html = chosen_path.read_text(encoding="utf-8", errors="replace")
                rejected_html = rejected_path.read_text(encoding="utf-8", errors="replace")

                obj = {
                    "id": f"{slide_id}",
                    "slide_id": slide_id,
                    "prompt": prompt,
                    "chosen": chosen_html,
                    "rejected": rejected_html,
                    "chosen_dir": chosen_dir,
                    "rejected_dir": rejected_dir,
                    "chosen_eiou_mean": chosen_score,
                    "rejected_eiou_mean": rejected_score,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                created += 1

    summary = {
        "total_slides": len(slides),
        "created_pairs": created,
        "skipped_missing_html": skipped_missing,
        "skipped_invalid_render_or_score": skipped_invalid,
        "skipped_tie": skipped_tie,
        "output": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

