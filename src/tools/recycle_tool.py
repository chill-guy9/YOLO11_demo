"""
安全删除：把文件/目录送进 Windows 回收站（可恢复），而不是永久删除
==================================================================
背景：Windows 上 rm/del 是永久删除，本项目清理历史产物必须可恢复。
      用 Shell API SHFileOperationW + FOF_ALLOWUNDO 走系统回收站，
      效果等同资源管理器里按 Delete。

用法：
  python src/recycle_tool.py --dry-run  某个目录 另一个目录      # 只看大小，不删
  python src/recycle_tool.py           某个目录 另一个目录      # 送回收站
  python src/recycle_tool.py --list-recycle                    # 查看回收站最近条目
"""
import argparse
import ctypes
import struct
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from ctypes import wintypes


class SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        ("pFrom", wintypes.LPCWSTR),
        ("pTo", wintypes.LPCWSTR),
        ("fFlags", ctypes.c_uint16),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", wintypes.LPCWSTR),
    ]


FO_DELETE = 3
FOF_ALLOWUNDO = 0x0040
FOF_NOCONFIRMATION = 0x0010
FOF_SILENT = 0x0004
FOF_NOERRORUI = 0x0400


def to_recycle(paths):
    """把一批路径送进回收站；返回 (rc, 实际提交的路径列表)"""
    existing = [str(Path(p).resolve()) for p in paths if Path(p).exists()]
    missing = [str(p) for p in paths if not Path(p).exists()]
    for m in missing:
        print(f"  [跳过] 不存在: {m}")
    if not existing:
        return 0, []
    joined = "\0".join(existing) + "\0\0"
    buf = ctypes.create_unicode_buffer(joined, len(joined))
    op = SHFILEOPSTRUCTW()
    op.hwnd = None
    op.wFunc = FO_DELETE
    op.pFrom = ctypes.cast(buf, wintypes.LPCWSTR)
    op.pTo = None
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return rc, existing


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def size_of(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def list_recycle(limit=10):
    root = Path("D:/$RECYCLE.BIN")
    tz = timezone(timedelta(hours=8))
    recs = []
    if not root.exists():
        print("未找到 D:\\$RECYCLE.BIN")
        return
    for sid in root.iterdir():
        if not sid.is_dir():
            continue
        for f in sid.glob("$I*"):
            try:
                b = f.read_bytes()
                if len(b) < 24:
                    continue
                ver = struct.unpack("<Q", b[0:8])[0]
                size = struct.unpack("<Q", b[8:16])[0]
                ft = struct.unpack("<Q", b[16:24])[0]
                dt = datetime.fromtimestamp((ft - 116444736000000000) / 10_000_000, tz)
                if ver == 2:
                    n = struct.unpack("<I", b[24:28])[0]
                    path = b[28:28 + n * 2].decode("utf-16-le", errors="ignore").rstrip("\x00")
                else:
                    path = b[24:24 + 520].decode("utf-16-le", errors="ignore").rstrip("\x00")
                recs.append((dt, size, path))
            except Exception:
                pass
    recs.sort(key=lambda x: x[0], reverse=True)
    print(f"回收站条目 {len(recs)} 条，最近 {limit} 条：")
    for dt, s, p in recs[:limit]:
        print(f"  {dt:%m-%d %H:%M:%S}  {human(s):>9}  {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="要删除的文件/目录")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-recycle", action="store_true")
    args = ap.parse_args()

    if args.list_recycle:
        list_recycle()
        return

    if not args.targets:
        raise SystemExit("请提供要删除的路径，或使用 --list-recycle")

    total = 0
    rows = []
    for t in args.targets:
        p = Path(t)
        if not p.exists():
            print(f"[跳过] 不存在: {t}")
            continue
        s = size_of(p)
        total += s
        rows.append((t, s))

    print(f"待处理 {len(rows)} 项，合计 {human(total)}：")
    for t, s in rows:
        print(f"  {human(s):>9}  {t}")
    if args.dry_run:
        print("\n[dry-run] 未执行删除")
        return

    print("\n正在送往回收站 ...")
    rc, submitted = to_recycle([t for t, _ in rows])
    print(f"SHFileOperation rc={rc}（0=成功），已提交 {len(submitted)} 项")
    if rc != 0:
        print("  rc 非 0 时需人工核对；但 Windows 上即使返回 2 也可能已成功，请用 --list-recycle 复查")
    left = [t for t, _ in rows if Path(t).exists()]
    if left:
        print(f"[!] 仍在原处 {len(left)} 项:")
        for t in left:
            print("   -", t)
    else:
        print("原处已全部清空")


if __name__ == "__main__":
    main()
