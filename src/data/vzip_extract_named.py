#!/usr/bin/env python3
"""
vzip_extract_named.py - extract a Star Fox (Eclipse engine) data.vzip with real file names.

Single self-contained script (no other files needed). Optional speed-up: pip install lz4

How it works (reverse-engineered format):
  * The file is a ZIP with a "VZIP" preamble table. Each member's data is a chain of LZ4
    blocks: [u32 size][LZ4 block]..., every block using the previous 64 KB as its dictionary.
  * A member name (hex, no leading zeros) is the 64-bit FNV-1a hash of a path WITHOUT its
    leading slash.
  * Paths are recovered from (a) every "/data/..." string inside the vjson-1 JSON members
    (package lists, exports, ...) and (b) the source path embedded in VIPR binaries.
    Every match is self-validated by the 64-bit hash.
  * Members with no recoverable path go to _unnamed/ (Wwise audio mostly) or _meta/
    (root package-list JSONs, labelled by their contents).
"""
import argparse
import collections
import csv
import json
import mmap
import os
import re
import shutil
import struct
import sys
import zipfile

PROG = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "vzip_extract_named.py"

USAGE_TEXT = """\
Usage:
  python {p} <file.vzip> [-o OUTDIR] [options]

Options:
  -o, --out DIR     output folder (default: output)
  --dry-run         only build the name map (names_map.csv) and print a summary; extract nothing
  --limit N         extract only the first N members (quick test)
  --skip-audio      do not extract unnamed audio (RIFF/BKHD); also excluded from the % shown
  --only-named      extract only files whose real name was recovered
  --zip-only        ignore table records that have no ZIP member. By default they ARE processed:
                    most are exact copies of ZIP members (skipped automatically), but some are
                    unique data (e.g. raw GPU buffers) that exist only in the table
  -h, --help        show this help

Examples:
  python {p} data.vzip --dry-run                      # summary only, writes names_map.csv
  python {p} data.vzip -o output --limit 500          # test with 500 files
  python {p} data.vzip -o output --skip-audio         # full extraction, no unnamed audio
  python {p} data.vzip -o output                      # full extraction (~33 GB)

Tip: install the lz4 module for much faster extraction:  pip install lz4
""".format(p=PROG)


class UsageParser(argparse.ArgumentParser):
    """ArgumentParser that prints a clear error followed by the full usage text."""

    def error(self, message):
        sys.stderr.write("error: %s\n\n%s" % (message, USAGE_TEXT))
        sys.exit(2)


def long_path(p):
    """On Windows, return the extended-length form (\\\\?\\...) so paths over 259 characters work."""
    if os.name != "nt":
        return p
    p = os.path.abspath(p)
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):                      # UNC share
        return "\\\\?\\UNC\\" + p[2:]
    return "\\\\?\\" + p


def write_file(path, data):
    """Create folders and write the file. Uses long-path form on Windows."""
    lp = long_path(path)
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    with open(lp, "wb") as fo:
        fo.write(data)


def die(message):
    sys.stderr.write("error: %s\n\n%s" % (message, USAGE_TEXT))
    sys.exit(2)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
M64 = (1 << 64) - 1
FNV_OFF, FNV_PRIME = 0xcbf29ce484222325, 0x100000001b3
HDR_SIZE = 12
REC = struct.Struct("<QQQQII")  # P, Q, comp, uncomp, flag, M  (40 bytes)
CHUNK = 65536
WINDOW = 65536
JSON_HEAD = b'{\n\t"'
SCAN = 8192
MAX_FULL = 16 << 20   # largest member that is fully decoded just to look for an embedded path
VPATH_RE = re.compile(rb'/(?:data|out)/[ -~]{3,400}')
BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f\\]')
PATH_RE = re.compile(rb'/[ -~]{7,400}')
STR_PATH = re.compile(rb'"(/(?:data|out)/[^"\\\n]{3,500})"')
MAGICS = [(b"\x89PNG", ".png"), (b"DDS ", ".dds"), (b"OggS", ".ogg"), (b"RIFF", ".wem"),
          (b"BKHD", ".bnk"), (b"VIPR", ".vipr"), (b"BNTX", ".bntx"), (b"FRES", ".bfres"),
          (b"SARC", ".sarc"), (b"Yaz0", ".yaz0"), (b"\x28\xb5\x2f\xfd", ".zst"),
          (b"PK\x03\x04", ".zip"), (b"<?xml", ".xml"), (b"\x1bLua", ".luac"),
          (b"\xff\xd8\xff", ".jpg"), (b"DXBC", ".dxbc"), (b"BNSH", ".bnsh")]


# --------------------------------------------------------------------------
# LZ4 (pure-Python fallback + optional fast module)
# --------------------------------------------------------------------------
class LZ4Error(Exception):
    pass


def lz4_block_decompress(src, dict_=b"", max_out=None):
    out = bytearray(dict_)
    base = len(out)
    i, n = 0, len(src)
    try:
        while i < n:
            tok = src[i]
            i += 1
            lit = tok >> 4
            if lit == 15:
                while True:
                    b = src[i]
                    i += 1
                    lit += b
                    if b != 255:
                        break
            if i + lit > n:
                raise LZ4Error("literals run past the end of the block")
            out += src[i:i + lit]
            i += lit
            if i >= n:
                break
            off = src[i] | (src[i + 1] << 8)
            i += 2
            if off == 0 or off > len(out):
                raise LZ4Error("invalid match offset %d" % off)
            ml = tok & 15
            if ml == 15:
                while True:
                    b = src[i]
                    i += 1
                    ml += b
                    if b != 255:
                        break
            ml += 4
            start = len(out) - off
            if off >= ml:
                out += out[start:start + ml]
            else:
                pat = bytes(out[start:])
                out += (pat * (ml // off + 1))[:ml]
            if max_out is not None and len(out) - base > max_out:
                raise LZ4Error("output larger than expected")
    except IndexError:
        raise LZ4Error("truncated block")
    return bytes(out[base:])


try:
    import lz4.block as _lz4b
    HAVE_LZ4 = True
except ImportError:
    _lz4b = None
    HAVE_LZ4 = False


def decode_block(src, want, dict_=b""):
    if HAVE_LZ4:
        try:
            return _lz4b.decompress(bytes(src), uncompressed_size=want, dict=dict_)
        except Exception:
            pass
    return lz4_block_decompress(src, dict_, max_out=want)


def decode_member(mm, off, comp, uncomp):
    """Chained blocks: [u32 size (high bit = stored raw)][block]..., 64 KB linked window."""
    pos, end = off, min(len(mm), off + comp)
    out = bytearray()
    while len(out) < uncomp:
        if pos + 4 > end:
            raise LZ4Error("data ended early")
        sz = struct.unpack_from("<I", mm, pos)[0]
        pos += 4
        raw = bool(sz & 0x80000000)
        sz &= 0x7FFFFFFF
        if sz == 0 or pos + sz > end:
            raise LZ4Error("invalid chunk size")
        blk = mm[pos:pos + sz]
        pos += sz
        want = min(CHUNK, uncomp - len(out))
        if raw:
            out += blk
        else:
            out += decode_block(blk, want, bytes(out[-WINDOW:]))
    if len(out) != uncomp:
        raise LZ4Error("final size mismatch")
    return bytes(out)


# --------------------------------------------------------------------------
# VZIP container
# --------------------------------------------------------------------------
Rec = collections.namedtuple("Rec", "idx P Q comp uncomp flag M")


class Vzip:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.size = os.fstat(self.f.fileno()).st_size
        if self.size < HDR_SIZE:
            die("file is too small to be a VZIP (needs at least %d bytes): %s" % (HDR_SIZE, path))
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        # Detection is by content only: the first 4 bytes must be 56 5A 49 50 ("VZIP").
        # The file extension is never checked.
        if self.mm[:4] != b"VZIP":
            die("not a VZIP file: the first 4 bytes are %s, expected 56 5A 49 50 ('VZIP'): %s"
                % (" ".join("%02X" % b for b in self.mm[:4]), path))
        self.count, self.extra = struct.unpack_from("<II", self.mm, 4)
        avail = (self.size - HDR_SIZE) // REC.size
        if self.count > avail:
            print("WARNING: header says %d records but only %d fit in the file (truncated download?)"
                  % (self.count, avail))
            self.count = avail
        end = HDR_SIZE + self.count * REC.size
        self.recs = [Rec(i, *t) for i, t in enumerate(REC.iter_unpack(self.mm[HDR_SIZE:end]))]


def read_member(v, r):
    """Decoded bytes of a record, or None on failure."""
    if r.Q < 0 or r.Q + r.comp > v.size:
        return None
    if r.flag == 0:
        return bytes(v.mm[r.Q:r.Q + r.comp]) if r.comp == r.uncomp else None
    try:
        d = decode_member(v.mm, r.Q, r.comp, r.uncomp)
    except (LZ4Error, ValueError):
        return None
    return d if len(d) == r.uncomp else None


def first_chunk(v, r):
    """First decoded chunk (up to 64 KB) of a record, or None."""
    off = r.Q
    if r.uncomp == 0:
        return b""
    if r.flag == 0:
        return bytes(v.mm[off:off + min(r.comp, CHUNK)])
    try:
        sz = struct.unpack_from("<I", v.mm, off)[0]
        raw = bool(sz & 0x80000000)
        sz &= 0x7FFFFFFF
        blk = v.mm[off + 4:off + 4 + sz]
        return bytes(blk) if raw else decode_block(blk, min(CHUNK, r.uncomp), b"")
    except Exception:
        return None


# --------------------------------------------------------------------------
# Naming helpers
# --------------------------------------------------------------------------
def fnv1a64(b):
    h = FNV_OFF
    for c in b:
        h = ((h ^ c) * FNV_PRIME) & M64
    return h


def fnv1a32(b):
    h = 0x811C9DC5
    for c in b:
        h = ((h ^ c) * 0x01000193) & 0xFFFFFFFF
    return h


def own_path(buf, own):
    """Find a '/...' string in buf whose hash (without the leading slash, any prefix) == own."""
    for m in PATH_RE.finditer(buf):
        s = m.group()
        h = FNV_OFF
        for i in range(1, len(s)):
            h = ((h ^ s[i]) * FNV_PRIME) & M64
            if h == own and i >= 6:
                return s[:i + 1].decode("ascii")
    return None


def describe(ps):
    """Label for unnamed JSONs: package_<dominant folder>_<number of paths>."""
    if len(ps) < 20:
        return "json_%d_paths" % len(ps)
    c = collections.Counter("_".join(p.split("/")[3:5]) for p in ps if p.count("/") >= 4)
    dom = c.most_common(1)[0][0] if c else "mixed"
    return "package_%s_%d" % (re.sub(r"[^A-Za-z0-9_.-]", "_", dom)[:50], len(ps))


def safe_path(p):
    parts = []
    for c in p.lstrip("/").split("/"):
        if c in ("", ".", ".."):
            continue
        c = BAD_CHARS.sub("_", c).rstrip(". ")
        parts.append(c or "_")
    return "/".join(parts)


def ext_for(d):
    for m, e in MAGICS:
        if d.startswith(m):
            return e
    if d[:2] in (b"{\n", b'{"', b"[{", b"[\n"):
        return ".json"
    return ".bin"


def json_asset_name(d):
    """Content-based name for a vjson-1 JSON: <path>.json (scripts: <path>.import.json)."""
    try:
        j = json.loads(d)
    except ValueError:
        return None
    guid, keys = None, ""
    for seg in j.get("segments", []):
        for o in seg.get("exported_objects", []):
            p = o.get("path")
            if isinstance(p, str) and p.startswith("/"):
                return p + (".import.json" if p.endswith(".vscript") else ".json")
            if guid is None:
                guid = str(o.get("$guid", ""))[:8]
                keys = "_".join(k for k in o if not k.startswith("$") and k != "vtable")[:40]
    return "_json_no_path/%s_%s.json" % (keys or "object", guid or "noguid")


def vipr_embedded_path(v, r, head):
    """First embedded '/data/...' path of a VIPR (prefers the '*.import' one); reads the whole
    member when the path is not inside the first 64 KB chunk."""
    def pick(buf):
        ps = [m.group().decode("ascii") for m in VPATH_RE.finditer(buf)]
        for p in ps:
            if p.endswith(".import"):
                return p
        return ps[0] if ps else None
    p = pick(head)
    if p is None and r.uncomp > len(head) and r.uncomp <= MAX_FULL:
        d = read_member(v, r)
        if d:
            p = pick(d)
    return p


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_parser():
    ap = UsageParser(add_help=False)
    ap.add_argument("file", nargs="?")
    ap.add_argument("-o", "--out", default="output")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-audio", action="store_true")
    ap.add_argument("--only-named", action="store_true")
    ap.add_argument("--zip-only", action="store_true")
    ap.add_argument("--all-records", action="store_true", help=argparse.SUPPRESS)  # kept for old command lines
    ap.add_argument("-h", "--help", action="store_true")
    return ap


def main():
    a = build_parser().parse_args()
    if a.help:
        print(USAGE_TEXT)
        return
    if not a.file:
        die("missing the input file")
    if not os.path.isfile(a.file):
        die("file not found: %s" % a.file)
    if a.limit < 0:
        die("--limit must be 0 or greater")
    a.all_records = not a.zip_only

    v = Vzip(a.file)
    try:
        z = zipfile.ZipFile(a.file)
    except zipfile.BadZipFile as e:
        die("could not read the ZIP directory at the end of the file (%s). "
            "Is the file complete (not truncated)?" % e)
    if not HAVE_LZ4:
        print("NOTE: 'lz4' module not found - using the slow pure-Python decoder. "
              "Install it with:  pip install lz4")

    # Several table records can point to the same data block (same Q): identical buffers are stored
    # once in the ZIP. The ZIP member belongs to the record whose M equals FNV-1a32 of the member
    # name; the other records at that Q are aliases with their own names and are handled as extras.
    by_q = collections.defaultdict(list)
    for r in v.recs:
        by_q[r.Q].append(r)
    matched = []
    shared_q = 0
    for zi in z.infolist():
        h = v.mm[zi.header_offset:zi.header_offset + 30]
        if h[:4] != b"PK\x03\x04":
            continue
        n, m = struct.unpack("<HH", h[26:30])
        cands = by_q.get(zi.header_offset + 30 + n + m)
        if not cands:
            continue
        r = None
        if len(cands) > 1:
            shared_q += 1
            mm = fnv1a32(zi.filename.encode())
            r = next((c for c in cands if c.M == mm), None)
        if r is None:
            r = cands[-1]
        matched.append((zi.filename, r))
    print("ZIP members matched to table records:", len(matched))
    if shared_q:
        print("ZIP members whose data block is shared by several table records:", shared_q)
    used = {r.idx for _, r in matched}

    # ---- pass 1: classify members, harvest paths ----
    json_entries, selfnamed, kinds = {}, {}, collections.Counter()
    audio_names, meta_label, paths, nonhex = set(), {}, set(), set()
    for i, (name, r) in enumerate(matched):
        try:
            own = int(name, 16)
        except ValueError:
            nonhex.add(name)
            kinds["non_hex_name"] += 1
            continue
        head = first_chunk(v, r)
        if head is None:
            kinds["read_error"] += 1
        elif head.startswith(JSON_HEAD) and 2 <= r.uncomp <= 64 << 20:
            d = read_member(v, r)
            if d and b"vjson-1" in d:
                json_entries[name] = r
                kinds["json"] += 1
                ps = [p.decode("utf-8", "replace") for p in STR_PATH.findall(d)]
                paths.update(ps)
                meta_label[name] = describe(ps)
                continue
            kinds["other_json"] += 1
        elif head[:4] in (b"RIFF", b"BKHD"):
            kinds["audio"] += 1
            audio_names.add(name)
        else:
            kinds["vipr" if head[:4] == b"VIPR" else "other_bin"] += 1
            p = own_path(head, own)
            if not p and head[:4] == b"VIPR" and len(head) < r.uncomp <= MAX_FULL:
                d = read_member(v, r)
                if d:
                    p = own_path(d, own)
            if p:
                selfnamed[name] = p
        if (i + 1) % 10000 == 0:
            print("  scanned %d/%d" % (i + 1, len(matched)))
    print("member types:", dict(kinds))

    # ---- pass 1b: table records without a ZIP member (--all-records): name them by content ----
    unmatched, rec_dest, extra_kind = [], {}, {}
    if a.all_records:
        unmatched = [("rec_%08d" % r.idx, r) for r in v.recs if r.idx not in used]
        print("table records without a ZIP member:", len(unmatched))
        k2 = collections.Counter()
        for i, (name, r) in enumerate(unmatched):
            head = first_chunk(v, r)
            if head is None:
                k2["read_error"] += 1
            elif head.startswith(JSON_HEAD) and 2 <= r.uncomp <= 64 << 20:
                d = read_member(v, r)
                if d and b"vjson-1" in d:
                    k2["json"] += 1
                    extra_kind[name] = "json"
                    paths.update(p.decode("utf-8", "replace") for p in STR_PATH.findall(d))
                    rel = json_asset_name(d)
                    if rel:
                        rec_dest[name] = rel
                else:
                    k2["other_json"] += 1
            elif head[:4] in (b"RIFF", b"BKHD"):
                k2["audio"] += 1
                extra_kind[name] = "audio"
                audio_names.add(name)
            elif head[:4] == b"VIPR":
                k2["vipr"] += 1
                extra_kind[name] = "vipr"
                p = vipr_embedded_path(v, r, head)
                if p:
                    rec_dest[name] = p + ".vipr"
                else:
                    k2["vipr_without_path"] += 1
            else:
                k2["other_bin"] += 1
                extra_kind[name] = "bin"
            if (i + 1) % 10000 == 0:
                print("  scanned %d/%d extra records" % (i + 1, len(unmatched)))
        print("extra record types:", dict(k2))
    hmap = {fnv1a64(p.lstrip("/").encode()): p for p in paths}
    print("vjson-1 files: %d | distinct paths: %d | binaries with a verified embedded path: %d"
          % (len(json_entries), len(paths), len(selfnamed)))

    # ---- table field M = FNV-1a 32 of the member name written in hex (verified on 98.5% of records) ----
    # So every extra record has a hidden 64-bit name. Hash every known path and look for records
    # whose M matches: this names raw buffers (.nvn_gen2.raw ...) that exist in the table but not in the ZIP.
    m_named = collections.defaultdict(list)       # fnv1a32(hex name) -> ZIP member names
    for name, _ in matched:
        if name not in nonhex:
            m_named[fnv1a32(name.encode())].append(name)
    via_hash = 0
    if unmatched:
        need = collections.defaultdict(list)
        for name, r in unmatched:
            if name not in rec_dest:
                need[r.M].append(name)
        if need:
            for p in paths:
                for cand, suffix in ((p, ""), (p + ".import", ".vipr")):
                    n64 = fnv1a64(cand.lstrip("/").encode())
                    for rname in need.get(fnv1a32(("%x" % n64).encode()), ()):
                        if rname not in rec_dest:
                            rec_dest[rname] = cand + suffix
                            via_hash += 1
        print("extra records named through their M hash: %d" % via_hash)

    # ---- final name map ----
    dest, seen, owner = {}, set(), {}   # owner: lower-case path -> (member name, record)

    def claim(rel, name, rec=None):
        key = rel.lower()
        if key in seen or len(rel) > 230:
            rel = "_long_or_duplicate/%s%s" % (name, os.path.splitext(rel)[1])
            key = rel.lower()
        seen.add(key)
        if rec is not None:
            owner[key] = (name, rec)
        return rel

    for name, r in matched:
        if name in nonhex:
            dest[name] = claim(safe_path(name), name, r)
            continue
        own = int(name, 16)
        if name in json_entries:
            if own in hmap:
                dest[name] = claim(safe_path(hmap[own]) + ".json", name, r)
        elif name in selfnamed:
            dest[name] = claim(safe_path(selfnamed[name]) + ".vipr", name, r)
        elif own in hmap:
            dest[name] = claim(safe_path(hmap[own]), name, r)

    # Extra records are mostly second copies of ZIP members. A copy is detected when
    #  (a) its content-derived path is already taken by a member with the same compressed and
    #      uncompressed size (JSON / VIPR), or
    #  (b) for binaries/audio, a ZIP member has the same sizes and identical first 64 KB chunk.
    dups = {}   # extra record name -> ZIP member name it duplicates
    by_name = dict(matched)
    by_size = collections.defaultdict(list)
    if unmatched:
        for name, r in matched:
            by_size[(r.comp, r.uncomp)].append((name, r))
    for name, r in unmatched:
        kind = extra_kind.get(name)
        if name in rec_dest:
            rel = safe_path(rec_dest[name])
            hit = owner.get(rel.lower())
            if hit and (hit[1].comp, hit[1].uncomp) == (r.comp, r.uncomp):
                dups[name] = hit[0]
                continue
            if hit:                      # same path, different data: keep it as a variant
                rel = "_variants/" + rel
            dest[name] = claim(rel, name, r)
        if name in dups or name in dest:
            continue
        if kind in ("bin", "audio") and name not in rec_dest:
            twin = next((c for c in m_named.get(r.M, []) if (by_name[c].comp, by_name[c].uncomp) == (r.comp, r.uncomp)), None)
            if twin:
                dups[name] = twin
                continue
            head = first_chunk(v, r)
            for cname, cr in by_size.get((r.comp, r.uncomp), [])[:20]:
                if head is not None and first_chunk(v, cr) == head:
                    dups[name] = cname
                    break
    for name in json_entries:
        if name not in dest:
            dest[name] = claim("_meta/%s_%s.json" % (meta_label.get(name, "json"), name[:8]), name)
    named = sum(1 for n in dest if not dest[n].startswith("_meta/"))
    total = len(matched) + len(unmatched) - len(dups)
    skipped_audio = 0
    if a.skip_audio:
        skipped_audio = sum(1 for n in audio_names if n not in dest and n not in dups)
        total -= skipped_audio
    if unmatched:
        by_kind = collections.Counter(extra_kind.get(n, "other") for n in dups)
        print("extra records that are copies of ZIP members (skipped, not counted): %d %s"
              % (len(dups), dict(by_kind)))
    print("members with a real name: %d of %d (%.1f%%)%s" % (
        named, total, 100.0 * named / max(1, total),
        " [excluding %d unnamed audio files because of --skip-audio]" % skipped_audio
        if a.skip_audio else ""))

    out_abs = os.path.abspath(a.out)
    if os.name == "nt" and len(out_abs) > 80:
        print("NOTE: the output folder path is long (%d characters). Paths over 259 characters are handled, "
              "but a short folder such as C:\\sf is safer." % len(out_abs))
    if "onedrive" in out_abs.lower():
        print("WARNING: the output folder is inside OneDrive. Syncing 100,000+ files (about 35 GB) can be "
              "very slow and may lock files; consider a folder outside OneDrive.")
    os.makedirs(a.out, exist_ok=True)
    map_path = os.path.join(a.out, "names_map.csv")
    with open(map_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "idx", "path", "comp", "uncomp", "duplicate_of"])
        for name, r in matched + unmatched:
            twin = dups.get(name, "")
            w.writerow([name, r.idx, dest.get(name, "") or (dest.get(twin, "") if twin else ""),
                        r.comp, r.uncomp, twin])
    print("name map written to", map_path)
    if a.dry_run:
        for n, p in list(dest.items())[:12]:
            print("  ", n, "->", p)
        print("(dry run: nothing was extracted)")
        return

    # ---- extraction ----
    work = list(matched) + unmatched
    if a.limit:
        work = work[:a.limit]

    def wanted(name):
        if name in dups:
            return False
        return name in dest or not (a.only_named or (a.skip_audio and name in audio_names))

    need = sum(r.uncomp for name, r in work if wanted(name))
    free = shutil.disk_usage(os.path.abspath(a.out)).free
    print("to extract: about %.1f GB | free space: %.1f GB" % (need / 1e9, free / 1e9))
    if free < need * 1.02:
        print("WARNING: there may not be enough free disk space in the output drive.")

    ok = bad = skipped = copies = long_used = 0
    for i, (name, r) in enumerate(work):
        if not wanted(name):
            if name in dups:
                copies += 1
            else:
                skipped += 1
            continue
        d = read_member(v, r)
        if d is None:
            bad += 1
            print("ERROR: could not decode member", name, "(record %d)" % r.idx)
            continue
        rel = dest.get(name)
        if rel is None:
            rel = claim("_unnamed/%s%s" % (name, ext_for(d[:16])), name)
        full = os.path.join(a.out, *rel.split("/"))
        if os.name == "nt" and len(os.path.abspath(full)) > 259:
            long_used += 1
        try:
            write_file(full, d)
        except OSError as e:
            # keep going: store the data under a short safe name and report the problem
            fb = os.path.join(a.out, "_write_failed", name + ext_for(d[:16]))
            try:
                write_file(fb, d)
                where = "saved as _write_failed/%s%s" % (name, ext_for(d[:16]))
            except OSError:
                where = "NOT saved"
            bad += 1
            print("ERROR writing %s: %s (%s)" % (rel, e.strerror or e, where))
            continue
        ok += 1
        if (i + 1) % 2000 == 0:
            print("  %d/%d" % (i + 1, len(work)))
    print("done: %d extracted, %d errors" % (ok, bad))
    if copies:
        print("not written: %d table records that are exact copies of files already written "
              "(see the duplicate_of column in names_map.csv)" % copies)
    if skipped:
        print("not written: %d records skipped by your options (--skip-audio / --only-named / --limit)" % skipped)
    if long_used:
        print("NOTE: %d files have paths longer than 259 characters (written with the Windows long-path form). "
              "Some programs, including File Explorer, may not open them; extracting to a short folder "
              "such as C:\\sf avoids this." % long_used)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted by user")
        sys.exit(130)
