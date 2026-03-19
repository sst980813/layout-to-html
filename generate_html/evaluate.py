import argparse
import json
import math
import statistics
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


def _lcs_len(a: str, b: str) -> int:
    """
    只计算 LCS 长度（字符级），用 O(min(n,m)) 内存。
    """
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for ch in long:
        cur = [0]
        for j, sj in enumerate(short, start=1):
            if ch == sj:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _ref_text_from_slide(slide: Dict[str, Any]) -> str:
    """
    参考文本：来自布局 JSON 的可见文本字段。
    - children: kind=="text" 的 content
    - 无 children: block.text_content
    """
    parts: List[str] = []
    for b in slide.get("blocks") or []:
        children = b.get("children") or []
        if children:
            for ch in children:
                if str(ch.get("kind") or "").lower() == "text":
                    t = str(ch.get("content") or "")
                    if t.strip():
                        parts.append(t)
        else:
            t = str(b.get("text_content") or "")
            if t.strip():
                parts.append(t)
    return "\n".join(parts).strip()


def _sanitize_filename(s: str) -> str:
    """与 generate.py 一致：将 slide_id 转为安全文件名。"""
    if not s:
        return ""
    for c in '\\/:*?"<>|':
        s = s.replace(c, "_")
    return s.strip() or "slide"


def _resolve_html_path(html_dir: Path, slide_id: str) -> Optional[Path]:
    """
    按 slide_id 查找对应 HTML，与 generate.py 一致：{slide_id}.html
    返回绝对路径。
    """
    p = (Path(html_dir) / f"{slide_id}.html").resolve()
    return p

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
  }

  const body = document.body;
  const bodyRect = body ? body.getBoundingClientRect() : null;
  const bodyVisible = !!(bodyRect && bodyRect.width > 0 && bodyRect.height > 0);
  const hasMain = !!document.querySelector('.slide') || bodyVisible;
  const visibleText = (document.body && document.body.innerText) ? document.body.innerText.trim() : '';
  const hasAnyContent = nodes.length > 0 || visibleText.length > 0 || !!document.querySelector('img');

  return {
    rectsById: out,
    hasMain,
    hasAnyContent,
    visibleText,
  };
})()
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="评测生成 HTML 的 VR / EIoU / CR（Playwright 渲染）")
    ap.add_argument("--input", default=str(Path("data/test_with_ids.json")), help="带 id 的布局 JSON（默认 data/test_with_ids.json）")
    ap.add_argument("--html-dir", default=str(Path("generate_html/output/rule")), help="生成 HTML 所在目录（默认 generate_html/output/llm）")
    ap.add_argument("--limit", type=int, default=None, help="只评测前 N 条（调试用）")
    ap.add_argument("--timeout-ms", type=int, default=8000, help="单页加载超时（默认 8000ms）")
    ap.add_argument("--headless", action="store_true", help="强制无头（默认即无头）")
    ap.add_argument("--show-per-slide", action="store_true", help="输出逐页指标")
    args = ap.parse_args()

    if sync_playwright is None:
        raise RuntimeError("未安装 playwright。请先 `pip install playwright` 并执行 `playwright install chromium`。")

    slides: List[Dict[str, Any]] = json.loads(Path(args.input).read_text(encoding="utf-8"))
    html_dir = Path(args.html_dir)
    if args.limit is not None:
        slides = slides[: max(0, args.limit)]

    vr_flags: List[int] = []
    all_eious: List[float] = []
    cr_scores: List[float] = []
    missing_html: List[int] = []
    per_slide_rows: List[Dict[str, Any]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page = context.new_page()

        # 捕获浏览器侧错误（用于 VR）
        page_errors: List[str] = []

        def _on_page_error(e: Any) -> None:
            try:
                page_errors.append(str(e))
            except Exception:
                page_errors.append("pageerror")

        page.on("pageerror", _on_page_error)

        for i, slide in enumerate(slides):
            page_errors.clear()
            html_path = _resolve_html_path(html_dir, slide.get("slide_id", ""))
            
            if html_path is None or not html_path.exists():
                # missing 不参与 validity_rate / cr 计算，只记缺失
                missing_html.append(i)
                per_slide_rows.append(
                    {
                        "index": i,
                        "slide_id": slide.get("slide_id", ""),
                        "valid": 0,
                        "eiou_mean": None,
                        "cr": 0.0,
                        "matched": 0,
                        "targets": sum(1 for _ in _iter_eval_targets_from_slide(slide)),
                        "html": None,
                        "error": "missing_html",
                    }
                )
                continue

            valid = 1
            extract: Dict[str, Any] = {}
            error_msg: Optional[str] = None
            try:
                page.goto(html_path.as_uri(), wait_until="load", timeout=args.timeout_ms)
                extract = page.evaluate(_JS_EXTRACT_RECTS)
                has_main = bool(extract.get("hasMain"))
                has_any = bool(extract.get("hasAnyContent"))
                if page_errors:
                    valid = 0
                    error_msg = f"pageerror: {page_errors[0]}"
                elif not (has_main and has_any):
                    valid = 0
                    error_msg = "no_visible_main_content"
            except PlaywrightError as e:
                valid = 0
                error_msg = f"playwright_error: {e}"
            except Exception as e:  # pragma: no cover
                valid = 0
                error_msg = f"error: {e}"

            vr_flags.append(valid)

            # CR：用 body.innerText 与参考文本做 LCS 覆盖率（以参考为分母）
            ref_text = _ref_text_from_slide(slide)
            gen_text = str(extract.get("visibleText") or "").strip()
            lcs = _lcs_len(gen_text, ref_text)
            cr = float(lcs / max(1, len(ref_text)))
            cr_scores.append(cr)

            # EIoU：按 id 匹配 target bbox 与渲染 bbox
            rects_by_id = extract.get("rectsById") or {}
            slide_eious: List[float] = []
            matched = 0
            targets = 0
            for json_id, tgt in _iter_eval_targets_from_slide(slide):
                targets += 1
                pr = rects_by_id.get(json_id)
                if not pr:
                    continue
                pred = Rect(float(pr["x"]), float(pr["y"]), float(pr["w"]), float(pr["h"]))
                matched += 1
                slide_eious.append(_iou(tgt, pred))
            all_eious.extend(slide_eious)

            per_slide_rows.append(
                {
                    "index": i,
                    "slide_id": slide.get("slide_id", ""),
                    "valid": valid,
                    "eiou_mean": (statistics.mean(slide_eious) if slide_eious else None),
                    "cr": cr,
                    "matched": matched,
                    "targets": targets,
                    "html": str(html_path),
                    "error": error_msg,
                }
            )

        context.close()
        browser.close()

    vr = float(sum(vr_flags) / max(1, len(vr_flags)))
    eiou = float(statistics.mean(all_eious)) if all_eious else 0.0
    cr_mean = float(statistics.mean(cr_scores)) if cr_scores else 0.0

    summary = {
        "count": len(slides),
        "missing_html": len(missing_html),
        "validity_rate": vr,
        "element_iou_mean": eiou,
        "content_retention_mean": cr_mean,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.show_per_slide:
        print(json.dumps(per_slide_rows, ensure_ascii=False, indent=2))

    # 若存在缺失或异常，给出非零退出码，方便脚本化调用
    if missing_html:
        return 2
    if any(v == 0 for v in vr_flags):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
