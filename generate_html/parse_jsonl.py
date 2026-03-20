import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Non-object JSON at line {line_no}: type={type(obj).__name__}")
            yield line_no, obj


def _safe_preview_record(
    obj: Dict[str, Any],
    *,
    preview_html: bool,
    max_html_chars: int,
) -> Dict[str, Any]:
    """
    为了避免把 chosen/rejected 这种超长 HTML 打满控制台：
    - 默认仅输出字段类型/长度/前缀（不输出完整 HTML）
    """
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, str) and (("chosen" in k) or ("rejected" in k) or "html" in k or k in {"prompt"}):
            if preview_html:
                out[k] = v[:max_html_chars] + ("..." if len(v) > max_html_chars else "")
            else:
                out[k] = {"type": "str", "len": len(v)}
        else:
            out[k] = v
    return out


def _collect_stats(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    key_counter: Counter = Counter()
    type_counter_by_key: Dict[str, Counter] = defaultdict(Counter)
    for obj in records:
        for k, v in obj.items():
            key_counter[k] += 1
            type_counter_by_key[k][type(v).__name__] += 1
    return {
        "records_examined": len(records),
        "top_keys": key_counter.most_common(30),
        "type_by_key_examples": {
            k: type_counter_by_key[k].most_common(5) for k in list(type_counter_by_key.keys())[:30]
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse and summarize a .jsonl file")
    ap.add_argument("--input", required=True, help="JSONL 文件路径")
    ap.add_argument("--head", type=int, default=0, help="打印前 N 条记录（默认 0，不打印内容）")
    ap.add_argument("--preview-html", action="store_true", help="打印 chosen/rejected/prompt 的内容片段（默认仅打印长度）")
    ap.add_argument("--max-html-chars", type=int, default=500, help="预览 HTML/文本最多字符数（默认 500）")
    ap.add_argument("--summary-only", action="store_true", help="只打印统计，不打印 head")
    args = ap.parse_args()

    path = Path(args.input).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    total = 0
    errors: List[str] = []
    sample_records: List[Dict[str, Any]] = []

    # 为了避免内存占用，head 以外只做统计，不保存全部记录
    # head 保存最多 N 条
    head_keep = int(args.head) if args.head and not args.summary_only else 0

    try:
        for line_no, obj in _iter_jsonl(path):
            total += 1
            if head_keep > 0 and len(sample_records) < head_keep:
                sample_records.append(obj)
    except ValueError as e:
        errors.append(str(e))

    print(json.dumps(
        {
            "file": str(path),
            "total_records": total,
            "parse_errors": errors,
        },
        ensure_ascii=False,
        indent=2,
    ))

    if sample_records:
        stats = _collect_stats(sample_records)
        print("=== Head sample stats ===")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

        for i, obj in enumerate(sample_records, start=1):
            print(f"=== Record {i} ===")
            preview = _safe_preview_record(
                obj,
                preview_html=bool(args.preview_html),
                max_html_chars=int(args.max_html_chars),
            )
            print(json.dumps(preview, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

