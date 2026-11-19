#!/usr/bin/env python3
"""
vipr_tool.py - converts .import.vipr + .nvn_gen2.raw assets back to common formats.

  textures : <name>.<png|tga>.import.vipr  + .N.nvn_gen2.raw  ->  <name>.<png|tga>
  models   : <name>.fbx.import.vipr        + .N.nvn_gen2.raw  ->  <name>.fbx (binary 7.4, with skeleton and skin)
  collision: <name>.collision.fbx.import.vipr (Embedded Bullet) ->  <name>.collision.fbx

Usage:
  python vipr_tool.py <file.import.vipr | folder> [-o output] [--lod N | --all-lods] [--mip N] [--no-rig] [--uv auto|tile|udim|raw] [--uv-grid U,V] [--uv-pad P] [--no-flip-v]
                          [--no-exact-uv] [--compare-uv]
  python vipr_tool.py info <file.import.vipr>

Dependencies: numpy, pillow, texture2ddecoder for BC1/BC3/BC4/BC5/BC7/ASTC; BC2 is decoded internally.
"""
import argparse, os, re, struct, sys
import numpy as np

MAGIC = b"VIPR"

class Vipr:
    def __init__(self, path):
        self.path = path
        self.data = open(path, "rb").read()
        d = self.data
        if d[:4] != MAGIC:
            raise ValueError(f"{path}: not a VIPR file")
        self.version = struct.unpack_from("<I", d, 4)[0]
        self.n_buffers = struct.unpack_from("<I", d, 0x1C)[0]
        if not d.endswith(b"GARDBYE!"):
            print(f"warning: {os.path.basename(path)} does not end with GARDBYE!", file=sys.stderr)
        self.is_texture = b"gpu-texture-nvn-gen2" in d
        self.is_model = (not self.is_texture) and b"gpu-nvn-gen2" in d
        self.is_bullet = b"BULLET" in d and b"SDNANAME" in d and not self.is_texture and not self.is_model
        paths = re.findall(rb"/data/[\x20-\x7e]+", d)
        paths = [p.decode() for p in paths if not p.endswith(b".import")]
        self.source_path = paths[0] if paths else None
        fn = os.path.basename(path)
        self.base = fn[:-len(".import.vipr")] if fn.endswith(".import.vipr") else os.path.splitext(fn)[0]

    def raw_path(self, n):
        return os.path.join(os.path.dirname(self.path), f"{self.base}.{n}.nvn_gen2.raw")

    def raw_exists(self, n):
        c = self.__dict__.setdefault("_rex", {})
        if n not in c: c[n] = os.path.exists(self.raw_path(n))
        return c[n]

    def raw_files(self):
        return [self.raw_path(n) for n in range(self.n_buffers) if os.path.exists(self.raw_path(n))]

    def buffer_sizes(self):
        """Buffer table: 32 B entries [.., 0xBFFF0000, idx, 0xCFFF0000, 0, 0, size, 2]."""
        if getattr(self, "_sizes", None) is not None: return self._sizes
        d, out = self.data, []
        for m in re.finditer(rb"\x00\x00\xff\xbf", d):
            s = m.start() - 4
            if s >= 0 and s + 32 <= len(d):
                v = struct.unpack_from("<8I", d, s)
                if v[3] == 0xCFFF0000 and v[7] == 2: out.append(v[6])
        self._sizes = out
        return out

VKFMT = {
    131: ("BC1", 8), 132: ("BC1", 8), 133: ("BC1", 8), 134: ("BC1", 8),
    135: ("BC2", 16), 136: ("BC2", 16),
    137: ("BC3", 16), 138: ("BC3", 16),
    139: ("BC4", 8), 140: ("BC4", 8),
    141: ("BC5", 16), 142: ("BC5", 16),
    145: ("BC7", 16), 146: ("BC7", 16),
    157: ("ASTC4x4", 16), 158: ("ASTC4x4", 16),
}

def cdiv(a, b):
    return (a + b - 1) // b

def align(a, b):
    return cdiv(a, b) * b

def initial_block_height(height_blocks):
    bh = 16
    while bh > 1 and height_blocks <= (bh // 2) * 8:
        bh //= 2
    return bh

def mip_layout(width, height, mip_count, bytes_per_block):
    out, offset = [], 0
    block_height = initial_block_height(cdiv(height, 4))
    for mip in range(mip_count):
        mw, mh = max(1, width >> mip), max(1, height >> mip)
        wb, hb = cdiv(mw, 4), cdiv(mh, 4)
        while block_height > 1 and hb <= (block_height // 2) * 8:
            block_height //= 2
        pitch = align(wb * bytes_per_block, 64)
        padded_h = align(hb, 8 * block_height)
        size = pitch * padded_h
        out.append((offset, mw, mh, wb, hb, block_height, size))
        offset += size
    return out

def deswizzle(data, width_bytes, height_blocks, block_height):
    """Unswizzle Switch 2 NVN block-linear data (address = f(y) + g(x), built by broadcast)."""
    pitch = align(width_bytes, 64)
    bh = block_height
    padded_height = align(height_blocks, 8 * bh)
    required = pitch * padded_height
    if len(data) < required:
        raise ValueError(f"raw data is too short for this mip: {len(data)} bytes; expected at least {required} bytes")
    gob_columns = pitch // 64
    y = np.arange(height_blocks, dtype=np.int64)
    x = np.arange(width_bytes, dtype=np.int64)
    ay = ((y // (8 * bh)) * 512 * bh * gob_columns + ((y % (8 * bh)) // 8) * 512
          + ((y % 8) // 4) * 128 + ((y % 4) // 2) * 32 + (y % 2) * 16)
    ax = (x // 64) * 512 * bh + ((x % 64) // 32) * 256 + ((x % 32) // 16) * 64 + (x % 16)
    return np.take(np.frombuffer(data, dtype=np.uint8), ay[:, None] + ax[None, :])

def find_texture_descs(data):
    """Descriptors: [VkFormat u16][w u16][h u16][depth=1][mips][layers=1]."""
    found = []
    for m in re.finditer(rb"([\x83-\x9e])\x00(..)(..)\x01\x00(..)\x01\x00", data, re.S):
        fmt = m.group(1)[0]
        if fmt not in VKFMT:
            continue
        width = struct.unpack("<H", m.group(2))[0]
        height = struct.unpack("<H", m.group(3))[0]
        mips = struct.unpack("<H", m.group(4))[0]
        if (4 <= width <= 16384 and 4 <= height <= 16384 and 1 <= mips <= 15
                and mips <= max(width, height).bit_length()):
            found.append((fmt, width, height, mips, m.start()))
    return found

def decode_bc2(data, width, height):
    """Decode BC2/DXT3 blocks to BGRA bytes."""
    bx, by = cdiv(width, 4), cdiv(height, 4)
    blocks = np.frombuffer(data, dtype=np.uint8)
    expected = bx * by * 16
    if blocks.size < expected:
        raise ValueError(f"not enough BC2 data: {blocks.size} bytes; expected {expected} bytes")
    blocks = blocks[:expected].reshape(by, bx, 16)
    c0 = blocks[:, :, 8].astype(np.uint16) | (blocks[:, :, 9].astype(np.uint16) << 8)
    c1 = blocks[:, :, 10].astype(np.uint16) | (blocks[:, :, 11].astype(np.uint16) << 8)

    def unpack565(c):
        r = ((c >> 11) & 31).astype(np.uint16)
        g = ((c >> 5) & 63).astype(np.uint16)
        b = (c & 31).astype(np.uint16)
        return np.stack(((r * 255 + 15) // 31, (g * 255 + 31) // 63,
                         (b * 255 + 15) // 31), axis=-1).astype(np.uint16)

    p0, p1 = unpack565(c0), unpack565(c1)
    colors = np.stack((p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3), axis=2)
    color_bits = blocks[:, :, 12].astype(np.uint32)
    for i in range(1, 4):
        color_bits |= blocks[:, :, 12 + i].astype(np.uint32) << (8 * i)
    selectors = ((color_bits[:, :, None, None] >>
                  (2 * np.arange(16, dtype=np.uint32).reshape(1, 1, 4, 4))) & 3).astype(np.intp)
    rgb = colors[np.arange(by)[:, None, None, None],
                 np.arange(bx)[None, :, None, None], selectors]

    alpha_bits = np.zeros((by, bx), dtype=np.uint64)
    for i in range(8):
        alpha_bits |= blocks[:, :, i].astype(np.uint64) << (8 * i)
    alpha = (((alpha_bits[:, :, None, None] >>
               (4 * np.arange(16, dtype=np.uint64).reshape(1, 1, 4, 4))) & 15) * 17).astype(np.uint16)
    rgba = np.empty((by, bx, 4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = rgb[..., 2].astype(np.uint8)
    rgba[..., 1] = rgb[..., 1].astype(np.uint8)
    rgba[..., 2] = rgb[..., 0].astype(np.uint8)
    rgba[..., 3] = alpha.astype(np.uint8)
    return rgba.transpose(0, 2, 1, 3, 4).reshape(by * 4, bx * 4, 4)[:height, :width].tobytes()

def decode_mip(fmt, width, height, linear_data):
    from PIL import Image
    name = VKFMT[fmt][0]
    if name == "BC2":
        pixels = decode_bc2(linear_data, width, height)
    else:
        import texture2ddecoder as t2d
        if name == "ASTC4x4":
            pixels = t2d.decode_astc(linear_data, width, height, 4, 4)
        else:
            decoder = getattr(t2d, "decode_" + name.lower(), None)
            if decoder is None:
                raise NotImplementedError(f"{name} decoding is not available in texture2ddecoder")
            pixels = decoder(linear_data, width, height)
    return Image.frombytes("RGBA", (width, height), pixels, "raw", "BGRA")

def convert_texture(v, outdir, mip=0):
    raw_path = v.raw_path(0)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"missing .raw file for {v.path}")
    raw_size = os.path.getsize(raw_path)
    descs = find_texture_descs(v.data)
    chosen = None
    for fmt, width, height, mips, _ in descs:
        layout = mip_layout(width, height, mips, VKFMT[fmt][1])
        if sum(item[6] for item in layout) == raw_size:
            chosen = (fmt, width, height, mips)
            break
    if chosen is None:
        raise RuntimeError(f"no descriptor matches {os.path.basename(raw_path)} ({raw_size} bytes); "
                           f"descriptors: {[item[:4] for item in descs]}")
    fmt, width, height, mips = chosen
    if not 0 <= mip < mips:
        raise ValueError(f"mip {mip} is out of range; the texture has {mips} levels (0 to {mips - 1})")
    offset, mw, mh, wb, hb, bh, size = mip_layout(width, height, mips, VKFMT[fmt][1])[mip]
    with open(raw_path, "rb") as f:
        f.seek(offset); chunk = f.read(size)
    linear = deswizzle(chunk, wb * VKFMT[fmt][1], hb, bh).tobytes()
    image = decode_mip(fmt, mw, mh, linear)
    if VKFMT[fmt][0] == "BC1":
        image = image.convert("RGB")
    source = os.path.basename(v.source_path) if v.source_path else v.base
    ext = os.path.splitext(source)[1].lower()
    if ext not in (".png", ".tga"):
        source, ext = v.base, os.path.splitext(v.base)[1].lower()
    filename = source if mip == 0 else f"{os.path.splitext(source)[0]}_mip{mip}{ext}"
    output = os.path.join(outdir, filename)
    os.makedirs(outdir, exist_ok=True)
    if ext == ".tga": image.save(output, "TGA")
    else: image.save(output, "PNG", compress_level=PNG_LEVEL)
    print(f"[texture] {os.path.basename(v.path)} -> {output} ({mw}x{mh}, {VKFMT[fmt][0]}, mips={mips})")
    return output

PNG_LEVEL = 1   # zlib level for PNG (1 = fast; lossless at any level). --png-level N

_SIG = re.compile(rb"\xff\xff\xff\xff(.)(.)\x03(.)(....)", re.S)

class Mesh:
    pass

def parse_meshes(v):
    if getattr(v, "_meshes", None) is not None: return v._meshes
    d = v.data
    meshes = []
    for m in _SIG.finditer(d):
        p = m.start()
        if p < 24 or p + 52 > len(d): continue
        idx32 = struct.unpack_from("<I", d, p + 8)[0]
        if idx32 not in (0, 1): continue
        V, I = struct.unpack_from("<II", d, p + 12)
        if not (0 < V < 2_000_000 and 0 < I < 8_000_000 and I % 3 == 0): continue
        if struct.unpack_from("<I", d, p + 20)[0] != 0: continue
        offs = []
        for i in range(8):
            o = struct.unpack_from("<I", d, p + 24 + 4 * i)[0]
            if o == 0: break
            offs.append(o)
        w = 4 if idx32 else 2
        if not offs or not (I * w <= offs[0] < I * w + 16): continue
        mesh = Mesh()
        mesh.obb = np.array(struct.unpack_from("<6f", d, p - 24), dtype=np.float64)
        mesh.skinned = bool(d[p + 5]); mesh.material = d[p + 7]; mesh.idx32 = bool(idx32)
        mesh.V, mesh.I, mesh.offs = V, I, offs
        meshes.append(mesh)
    v._meshes = meshes
    return meshes

def global_box(meshes):
    mn = np.min([m.obb[:3] for m in meshes], axis=0)
    mx = np.max([m.obb[3:] for m in meshes], axis=0)
    return (mn + mx) / 2, (mx - mn) / 2

def resolve_raw(v, meshes, k, sizes=None):
    if v.raw_exists(k): return v.raw_path(k), False
    sizes = sizes if sizes is not None else v.buffer_sizes()
    m = meshes[k]
    for j, o in enumerate(meshes):
        if j != k and v.raw_exists(j) and (j >= len(sizes) or k >= len(sizes) or sizes[j] == sizes[k]) \
                and (o.V, o.I, o.offs) == (m.V, m.I, m.offs) and np.allclose(o.obb, m.obb):
            return v.raw_path(j), True
    return None, False

def _decode_octahedral(v):
    p = np.asarray(v, dtype=np.float64) * 2.0 - 1.0
    x, y = p[:, 0].copy(), p[:, 1].copy()
    z = 1.0 - np.abs(x) - np.abs(y)
    t = np.maximum(-z, 0.0)
    x += np.where(x >= 0.0, -t, t)
    y += np.where(y >= 0.0, -t, t)
    out = np.stack([x, y, z], axis=1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

def _mesh_binormals(pos, idx, uv, normal, tangent):
    n = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    t = tangent - np.sum(tangent * n, axis=1, keepdims=True) * n
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    tri = idx.astype(np.int64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = pos[b] - pos[a], pos[c] - pos[a]
    d1, d2 = uv[b] - uv[a], uv[c] - uv[a]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    face_n = np.cross(e1, e2)
    orient = np.sign(np.einsum("ij,ij->i", face_n, (n[a] + n[b] + n[c]) / 3.0))
    hand = np.sign(det) * np.where(orient != 0, orient, 1.0)
    weight = np.maximum(np.linalg.norm(face_n, axis=1), 1e-12)
    contrib = np.where(np.abs(det) > 1e-12, hand * weight, 0.0)
    V = len(pos)
    votes = (np.bincount(a, contrib, V) + np.bincount(b, contrib, V) + np.bincount(c, contrib, V))
    hv = np.where(votes < 0.0, -1.0, 1.0)
    return np.cross(n, t) * hv[:, None]

def load_mesh_data(v, k, mesh, gbox, path=None):
    with open(path or v.raw_path(k), "rb") as f: raw = f.read()
    mv = memoryview(raw)
    V, I = mesh.V, mesh.I
    bounds = mesh.offs + [len(raw)]
    streams = [mv[bounds[i]:bounds[i + 1]] for i in range(len(mesh.offs)) if bounds[i + 1] > bounds[i]]
    widths = [len(s) // V for s in streams]
    if widths not in ([8, 4, 8, 8], [8, 4, 8, 8, 4, 4]) or any(len(s) != w * V for s, w in zip(streams, widths)):
        raise ValueError(f"unexpected vertex layout {widths}")
    idx = np.frombuffer(mv[:I * (4 if mesh.idx32 else 2)], dtype="<u4" if mesh.idx32 else "<u2").reshape(-1, 3).astype(np.int32)
    q = np.frombuffer(streams[0], dtype="<i2").reshape(V, 4)[:, :3].astype(np.float64) / 32767.0
    pos = gbox[0] + q * gbox[1]
    uvr = np.frombuffer(streams[2], dtype="<i2").reshape(V, 4)[:, :2].astype(np.float64) / 32767.0
    nt = np.frombuffer(streams[3], dtype="<f2").reshape(V, 4).astype(np.float64)
    normal, tangent = _decode_octahedral(nt[:, :2]), _decode_octahedral(nt[:, 2:])
    skin = (np.frombuffer(streams[5], dtype=np.uint8).reshape(V, 4), np.frombuffer(streams[4], dtype=np.uint8).reshape(V, 4)) \
        if len(streams) == 6 else None
    return idx, pos, uvr, skin, normal, tangent

def _fit_uv_offset(mesh_data, normal_map_path):
    from PIL import Image
    image = np.asarray(Image.open(normal_map_path).convert("RGB"), dtype=np.float64) / 255.0
    h, w = image.shape[:2]
    prepared = []
    all_uv = []
    for idx, pos, uv, normal, tangent in mesh_data:
        tri = idx.astype(np.int64)
        all_uv.append(uv)
        n = normal[tri].mean(axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        t = tangent[tri].mean(axis=1)
        t -= np.sum(t * n, axis=1, keepdims=True) * n
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        face = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
        area = np.linalg.norm(face, axis=1)
        face /= np.maximum(area[:, None], 1e-12)
        orient = np.sign(np.sum(face * n, axis=1))
        orient = np.where(orient == 0.0, 1.0, orient)
        face *= orient[:, None]
        d1, d2 = uv[tri[:, 1]] - uv[tri[:, 0]], uv[tri[:, 2]] - uv[tri[:, 0]]
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        hand = np.sign(det) * orient
        b = np.cross(n, t) * hand[:, None]
        expected = np.stack((np.sum(face * t, axis=1), np.sum(face * b, axis=1),
                             np.sum(face * n, axis=1)), axis=1)
        expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1e-12)
        prepared.append((uv[tri].mean(axis=1), expected, area))
    bounds = np.concatenate(all_uv, axis=0)
    du_min, dv_min = np.maximum(-0.02, -bounds.min(axis=0))
    du_max, dv_max = np.minimum(0.02, 1.0 - bounds.max(axis=0))

    def score(du, dv):
        total, weight_sum = 0.0, 0.0
        for uv_center, expected, area in prepared:
            q = uv_center + np.array([du, dv])
            x = np.clip(q[:, 0] * (w - 1), 0, w - 1)
            y = np.clip((1.0 - q[:, 1]) * (h - 1), 0, h - 1)
            x0, y0 = x.astype(np.int64), y.astype(np.int64)
            fx, fy = x - x0, y - y0
            x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
            nm = (image[y0, x0] * ((1-fx)*(1-fy))[:, None]
                  + image[y0, x1] * (fx*(1-fy))[:, None]
                  + image[y1, x0] * ((1-fx)*fy)[:, None]
                  + image[y1, x1] * (fx*fy)[:, None]) * 2.0 - 1.0
            nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-12)
            weights = area
            total += float(np.sum(np.sum(nm * expected, axis=1) * weights))
            weight_sum += float(np.sum(weights))
        return total / max(weight_sum, 1e-12)

    best = (-np.inf, (0.0, 0.0))
    for du in np.arange(du_min, du_max + 0.0005, 0.001):
        for dv in np.arange(dv_min, dv_max + 0.0005, 0.001):
            value = score(float(du), float(dv))
            if value > best[0]: best = (value, (float(du), float(dv)))
    du0, dv0 = best[1]
    lo_u, hi_u = max(du_min, du0 - 0.001), min(du_max, du0 + 0.001)
    lo_v, hi_v = max(dv_min, dv0 - 0.001), min(dv_max, dv0 + 0.001)
    for du in np.arange(lo_u, hi_u + 0.00005, 0.0001):
        for dv in np.arange(lo_v, hi_v + 0.00005, 0.0001):
            value = score(float(du), float(dv))
            if value > best[0]: best = (value, (float(du), float(dv)))
    return best[1], score(0.0, 0.0), best[0]

def _material_slots_inline(d):
    names = [(m.start(), m.group().decode()) for m in re.finditer(rb"m_[A-Za-z0-9_]+", d)
             if d[m.start() - 8:m.start() - 4] == b"\x00\x00\xff\xbf"
             and struct.unpack_from("<I", d, m.start() - 4)[0] == len(m.group())]
    if len(names) < 3: return None
    stride = 56
    if sum(1 for i in range(len(names) - 1) if names[i + 1][0] - names[i][0] == stride) < 2: return None
    ref = names[0][0] - 8
    inline = {(p - 8 - ref) // stride: n for p, n in names if (p - 8 - ref) % stride == 0}
    pool = [(m.start(), m.group().decode()) for m in re.finditer(rb"(?<=\x00)m_[A-Za-z0-9_]+(?=\x00)", d)]
    used = set(inline.values())
    out, unresolved = {}, []
    jmin = None
    for j in range(min(inline) - 6, max(inline) + 7):
        h = ref + stride * j
        if h < 0 or d[h:h + 4] != b"\x00\x00\xff\xbf": continue
        if jmin is None: jmin = j
        ln = struct.unpack_from("<I", d, h + 4)[0]
        if j in inline: out[j] = inline[j]
        elif ln >> 24:
            n = ln & 0xFFFFFF
            # only accept a name if it is the single unused candidate of that length;
            # with several candidates the choice would be a guess (swaps names between meshes)
            cand = [s for s in dict.fromkeys(s for _, s in pool) if len(s) == n and s not in used]
            if len(cand) == 1: out[j] = cand[0]; used.add(cand[0])
            else: unresolved.append(j - jmin)
        else: unresolved.append(j - jmin)
    if not out: return None
    # keep the real slot index (no compaction): a gap must not shift the following names
    last = max(out) - jmin
    res = [out.get(jmin + i, f"material_{i}") for i in range(last + 1)]
    if unresolved:
        print(f"warning: material slots without a reliable name (using material_N): {sorted(set(unresolved))}", file=sys.stderr)
    return res

def material_names(v):
    d = v.data
    slots = _material_slots_inline(d)
    if slots: return slots, ".vipr slot array"
    pool, seen = [], set()
    for m in re.finditer(rb"(?<=\x00)m_[A-Za-z0-9_]+(?=\x00)", d):
        n = m.group().decode()
        if n not in seen: seen.add(n); pool.append(n)
    jp = os.path.join(os.path.dirname(v.path), v.base.rsplit(".", 1)[0] + ".model.json")
    if pool and os.path.exists(jp):
        import json
        try:
            slots = [pool[0]] + pool[:0:-1]
            for seg in json.load(open(jp)).get("segments", []):
                for o in seg.get("exported_objects", []):
                    for mo in o.get("material_overrides", []):
                        s = mo.get("slot", 0)
                        if s < len(slots) and mo.get("name"): slots[s] = mo["name"]
            return slots, "model.json (slot -> name)"
        except Exception as e:
            print("warning: unreadable model.json:", e, file=sys.stderr)
    return pool, "string pool order (NOT verified)"

def lod_groups(meshes):
    out, seen, cur = [], set(), 0
    for m in meshes:
        if m.material in seen: cur += 1; seen = set()
        seen.add(m.material); out.append(cur)
    return out

_BONE = 164

def _find_bone_runs(d):
    n = len(d); cands = set()
    for sh in range(4):
        m = (n - sh) // 4
        L = m - 41
        if L <= 0: continue
        f = np.frombuffer(d, dtype="<f4", count=m, offset=sh); i = np.frombuffer(d, dtype="<i4", count=m, offset=sh)
        with np.errstate(all="ignore"):
            j = np.flatnonzero(np.abs(f[35:35 + L] - 1) < 1e-4)
            if not j.size: continue
            j = j[(i[2 + j] >= -1) & (i[2 + j] < 4096) & (i[3 + j] >= 0) & (i[3 + j] <= 7)]
            j = j[(np.abs(f[23 + j]) < 1e-3) & (np.abs(f[27 + j]) < 1e-3) & (np.abs(f[31 + j]) < 1e-3)]
            if not j.size: continue
            q = np.stack([f[7 + j], f[8 + j], f[9 + j], f[10 + j]], 1)
            j = j[np.abs(np.linalg.norm(q, axis=1) - 1) < 2e-3]
        cands.update(sh + 4 * int(x) for x in j)
    runs = []
    for s in sorted(cands):
        if s - _BONE in cands: continue
        r = [s]
        while r[-1] + _BONE in cands: r.append(r[-1] + _BONE)
        runs.append(r)
    return runs

def _quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

class Bone:
    pass

def parse_skeleton(v):
    d = v.data
    runs = _find_bone_runs(d)
    if not runs: return None
    run = max(runs, key=len)
    bones = []
    for s in run:
        b = Bone()
        b.ptr, = struct.unpack_from("<I", d, s)
        b.parent, = struct.unpack_from("<i", d, s + 8)
        b.pos = np.array(struct.unpack_from("<3f", d, s + 16), dtype=np.float64)
        b.quat = np.array(struct.unpack_from("<4f", d, s + 28), dtype=np.float64)
        b.S = np.array(struct.unpack_from("<9f", d, s + 44), dtype=np.float64).reshape(3, 3)
        b.inv = np.frombuffer(d[s + 80:s + 144], dtype="<f4").reshape(4, 4).astype(np.float64).T
        bones.append(b)
    if any(not (-1 <= b.parent < i) for i, b in enumerate(bones)):
        print("warning: inconsistent bone hierarchy; rig ignored", file=sys.stderr); return None
    P = {m.start() for m in re.finditer(rb"(?<![^\x00])[\x20-\x7e]+(?=\x00)", d)}
    base = next((p - bones[0].ptr for p in sorted(P) if p - bones[0].ptr >= 0 and all(p - bones[0].ptr + b.ptr in P for b in bones)), None)
    for i, b in enumerate(bones):
        if base is None: b.name = f"bone_{i}"
        else:
            o = base + b.ptr; b.name = d[o:d.index(b"\x00", o)].decode()
    for i, b in enumerate(bones):
        M = np.eye(4); M[:3, :3] = _quat_to_R(b.quat) @ b.S; M[:3, 3] = b.pos
        b.local = M
        b.world = M if b.parent < 0 else bones[b.parent].world @ M
    err = max(np.abs(np.linalg.inv(b.world) - b.inv).max() / max(1.0, np.abs(b.inv).max()) for b in bones)
    if err > 1e-3: print(f"warning: InverseWorldTransform does not match hierarchy (error {err:.1e})", file=sys.stderr)
    return bones

def _euler_xyz_deg(R):
    sy = -R[2, 0]; y = np.arcsin(np.clip(sy, -1, 1))
    if abs(sy) < 1 - 1e-8: x = np.arctan2(R[2, 1], R[2, 2]); z = np.arctan2(R[1, 0], R[0, 0])
    else: x = np.arctan2(-R[1, 2], R[1, 1]); z = 0.0
    return np.degrees([x, y, z])

class FbxNode:
    def __init__(self, name, props=(), children=()):
        self.name, self.props, self.children = name, list(props), list(children)
    def add(self, *nodes):
        self.children.extend(nodes); return self

def _enc_prop(p):
    t, val = p
    if t == "I": return b"I" + struct.pack("<i", val)
    if t == "L": return b"L" + struct.pack("<q", val)
    if t == "D": return b"D" + struct.pack("<d", val)
    if t == "C": return b"C" + struct.pack("<?", val)
    if t == "S":
        b = val.encode("utf-8"); return b"S" + struct.pack("<I", len(b)) + b
    if t == "d":
        a = np.ascontiguousarray(val, dtype="<f8"); return b"d" + struct.pack("<III", len(a), 0, a.nbytes) + a.tobytes()
    if t == "i":
        a = np.ascontiguousarray(val, dtype="<i4"); return b"i" + struct.pack("<III", len(a), 0, a.nbytes) + a.tobytes()
    raise ValueError(t)

def _write_node(out, n):
    start = len(out)
    out += b"\0" * 12
    nb = n.name.encode(); out += struct.pack("<B", len(nb)) + nb
    pstart = len(out)
    for p in n.props: out += _enc_prop(p)
    plen = len(out) - pstart
    if n.children:
        for c in n.children: _write_node(out, c)
        out += b"\0" * 13
    struct.pack_into("<III", out, start, len(out), len(n.props), plen)

def write_fbx(path, nodes):
    out = bytearray(b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7400))
    for n in nodes: _write_node(out, n)
    out += b"\0" * 13
    out += b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e" + b"\0" * 4
    pad = ((len(out) + 15) & ~15) - len(out)
    out += b"\0" * (pad or 16)
    out += struct.pack("<I", 7400) + b"\0" * 120
    out += b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
    open(path, "wb").write(out)

def _P(*a):
    return FbxNode("P", a)

def _vec_p(name, x, y, z):
    return _P(("S", name), ("S", name), ("S", ""), ("S", "A"), ("D", float(x)), ("D", float(y)), ("D", float(z)))

def _cluster_transform(bone):
    return np.linalg.inv(bone.world)

def _fbx_matrix(M):
    return np.asarray(M, dtype=np.float64).T.reshape(-1)

def build_fbx(items, bones=None, creator="vipr_tool"):
    uid = [1000000]
    def nid():
        uid[0] += 1; return uid[0]
    header = FbxNode("FBXHeaderExtension", children=[
        FbxNode("FBXHeaderVersion", [("I", 1003)]), FbxNode("FBXVersion", [("I", 7400)]),
        FbxNode("Creator", [("S", creator)])])
    gs = FbxNode("GlobalSettings", children=[
        FbxNode("Version", [("I", 1000)]),
        FbxNode("Properties70", children=[
            _P(("S", "UpAxis"), ("S", "int"), ("S", "Integer"), ("S", ""), ("I", 2)),
            _P(("S", "UpAxisSign"), ("S", "int"), ("S", "Integer"), ("S", ""), ("I", 1)),
            _P(("S", "FrontAxis"), ("S", "int"), ("S", "Integer"), ("S", ""), ("I", 1)),
            _P(("S", "FrontAxisSign"), ("S", "int"), ("S", "Integer"), ("S", ""), ("I", -1)),
            _P(("S", "CoordAxis"), ("S", "int"), ("S", "Integer"), ("S", ""), ("I", 0)),
            _P(("S", "CoordAxisSign"), ("S", "int"), ("S", "Integer"), ("S", ""), ("I", 1)),
            _P(("S", "UnitScaleFactor"), ("S", "double"), ("S", "Number"), ("S", ""), ("D", 100.0)),
            _P(("S", "OriginalUnitScaleFactor"), ("S", "double"), ("S", "Number"), ("S", ""), ("D", 100.0))])])
    objects, conns = FbxNode("Objects"), FbxNode("Connections")
    n_objs = 0

    bone_ids = []
    if bones:
        for b in bones:
            mid, aid = nid(), nid(); bone_ids.append(mid)
            rot = _euler_xyz_deg(b.local[:3, :3] / np.maximum(np.linalg.norm(b.local[:3, :3], axis=0), 1e-12))
            sc = np.linalg.norm(b.local[:3, :3], axis=0)
            objects.add(
                FbxNode("NodeAttribute", [("L", aid), ("S", f"{b.name}\x00\x01NodeAttribute"), ("S", "LimbNode")],
                        [FbxNode("TypeFlags", [("S", "Skeleton")])]),
                FbxNode("Model", [("L", mid), ("S", f"{b.name}\x00\x01Model"), ("S", "LimbNode")], [
                    FbxNode("Version", [("I", 232)]),
                    FbxNode("Properties70", children=[_vec_p("Lcl Translation", *b.local[:3, 3]),
                                                      _vec_p("Lcl Rotation", *rot), _vec_p("Lcl Scaling", *sc)]),
                    FbxNode("Shading", [("C", True)]), FbxNode("Culling", [("S", "CullingOff")])]))
            conns.add(FbxNode("C", [("S", "OO"), ("L", aid), ("L", mid)]),
                      FbxNode("C", [("S", "OO"), ("L", mid), ("L", bone_ids[b.parent] if b.parent >= 0 else 0)]))
            n_objs += 2

    mat_ids = {}
    for it in items:
        name, mat, pos, idx, uv = it[:5]
        skin = it[5] if len(it) > 5 else None
        normal = it[6] if len(it) > 6 else None
        tangent = it[7] if len(it) > 7 else None
        binormal = it[8] if len(it) > 8 else None
        if mat not in mat_ids:
            mid = nid(); mat_ids[mat] = mid
            objects.add(FbxNode("Material", [("L", mid), ("S", f"{mat}\x00\x01Material"), ("S", "")], [
                FbxNode("Version", [("I", 102)]), FbxNode("ShadingModel", [("S", "phong")]),
                FbxNode("MultiLayer", [("I", 0)])]))
            n_objs += 1
        gid, mdl = nid(), nid()
        poly = idx.copy().astype(np.int32)
        poly[:, 2] = -(poly[:, 2] + 1)
        layer_children = [FbxNode("Version", [("I", 100)])]
        geo_children = [FbxNode("GeometryVersion", [("I", 124)]),
                        FbxNode("Vertices", [("d", pos.reshape(-1))]),
                        FbxNode("PolygonVertexIndex", [("i", poly.reshape(-1))])]
        if uv is not None:
            geo_children.append(FbxNode("LayerElementUV", [("I", 0)], [
                FbxNode("Version", [("I", 101)]), FbxNode("Name", [("S", "UVMap")]),
                FbxNode("MappingInformationType", [("S", "ByVertice")]),
                FbxNode("ReferenceInformationType", [("S", "Direct")]),
                FbxNode("UV", [("d", uv.reshape(-1))])]))
            layer_children.append(FbxNode("LayerElement", children=[FbxNode("Type", [("S", "LayerElementUV")]), FbxNode("TypedIndex", [("I", 0)])]))
        if normal is not None:
            geo_children.append(FbxNode("LayerElementNormal", [("I", 0)], [
                FbxNode("Version", [("I", 101)]), FbxNode("Name", [("S", "Normals")]),
                FbxNode("MappingInformationType", [("S", "ByVertice")]),
                FbxNode("ReferenceInformationType", [("S", "Direct")]),
                FbxNode("Normals", [("d", normal.reshape(-1))])]))
            layer_children.append(FbxNode("LayerElement", children=[FbxNode("Type", [("S", "LayerElementNormal")]), FbxNode("TypedIndex", [("I", 0)])]))
        if tangent is not None:
            geo_children.append(FbxNode("LayerElementTangent", [("I", 0)], [
                FbxNode("Version", [("I", 101)]), FbxNode("Name", [("S", "Tangents")]),
                FbxNode("MappingInformationType", [("S", "ByVertice")]),
                FbxNode("ReferenceInformationType", [("S", "Direct")]),
                FbxNode("Tangents", [("d", tangent.reshape(-1))])]))
            layer_children.append(FbxNode("LayerElement", children=[FbxNode("Type", [("S", "LayerElementTangent")]), FbxNode("TypedIndex", [("I", 0)])]))
        if binormal is not None:
            geo_children.append(FbxNode("LayerElementBinormal", [("I", 0)], [
                FbxNode("Version", [("I", 101)]), FbxNode("Name", [("S", "Binormals")]),
                FbxNode("MappingInformationType", [("S", "ByVertice")]),
                FbxNode("ReferenceInformationType", [("S", "Direct")]),
                FbxNode("Binormals", [("d", binormal.reshape(-1))])]))
            layer_children.append(FbxNode("LayerElement", children=[FbxNode("Type", [("S", "LayerElementBinormal")]), FbxNode("TypedIndex", [("I", 0)])]))
        geo_children.append(FbxNode("LayerElementMaterial", [("I", 0)], [
            FbxNode("Version", [("I", 101)]), FbxNode("Name", [("S", "")]),
            FbxNode("MappingInformationType", [("S", "AllSame")]),
            FbxNode("ReferenceInformationType", [("S", "IndexToDirect")]),
            FbxNode("Materials", [("i", np.array([0]))])]))
        layer_children.append(FbxNode("LayerElement", children=[FbxNode("Type", [("S", "LayerElementMaterial")]), FbxNode("TypedIndex", [("I", 0)])]))
        geo_children.append(FbxNode("Layer", [("I", 0)], layer_children))
        objects.add(FbxNode("Geometry", [("L", gid), ("S", f"{name}\x00\x01Geometry"), ("S", "Mesh")], geo_children),
                    FbxNode("Model", [("L", mdl), ("S", f"{name}\x00\x01Model"), ("S", "Mesh")], [
                        FbxNode("Version", [("I", 232)]), FbxNode("Properties70"),
                        FbxNode("Shading", [("C", True)]), FbxNode("Culling", [("S", "CullingOff")])]))
        conns.add(FbxNode("C", [("S", "OO"), ("L", mdl), ("L", 0)]),
                  FbxNode("C", [("S", "OO"), ("L", gid), ("L", mdl)]),
                  FbxNode("C", [("S", "OO"), ("L", mat_ids[mat]), ("L", mdl)]))
        n_objs += 3
        if skin is not None and bones:
            bi, bw = skin
            V = len(pos)
            vid = np.repeat(np.arange(V), 4); bid = bi.reshape(-1).astype(np.int64); w = bw.reshape(-1).astype(np.float64) / 255.0
            keep = (w > 0) & (bid < len(bones)); vid, bid, w = vid[keep], bid[keep], w[keep]
            order = np.lexsort((vid, bid)); vid, bid, w = vid[order], bid[order], w[order]
            sid = nid()
            objects.add(FbxNode("Deformer", [("L", sid), ("S", f"{name}\x00\x01Deformer"), ("S", "Skin")], [
                FbxNode("Version", [("I", 101)]), FbxNode("Link_DeformAcuracy", [("D", 50.0)])]))
            conns.add(FbxNode("C", [("S", "OO"), ("L", sid), ("L", gid)]))
            n_objs += 1
            cuts = np.concatenate(([0], np.flatnonzero(np.diff(bid)) + 1, [len(bid)])) if len(bid) else np.zeros(1, dtype=np.int64)
            for s0, s1 in zip(cuts[:-1], cuts[1:]):
                b = int(bid[s0])
                vv, inv_ = np.unique(vid[s0:s1], return_inverse=True); ww = np.bincount(inv_, weights=w[s0:s1])
                cid = nid()
                objects.add(FbxNode("Deformer", [("L", cid), ("S", f"{bones[b].name}\x00\x01SubDeformer"), ("S", "Cluster")], [
                    FbxNode("Version", [("I", 100)]), FbxNode("UserData", [("S", ""), ("S", "")]),
                    FbxNode("Indexes", [("i", vv.astype(np.int32))]), FbxNode("Weights", [("d", ww)]),
                    FbxNode("Transform", [("d", _fbx_matrix(_cluster_transform(bones[b])))]),
                    FbxNode("TransformLink", [("d", _fbx_matrix(bones[b].world))])]))
                conns.add(FbxNode("C", [("S", "OO"), ("L", cid), ("L", sid)]),
                          FbxNode("C", [("S", "OO"), ("L", bone_ids[b]), ("L", cid)]))
                n_objs += 1
    defs = FbxNode("Definitions", children=[FbxNode("Version", [("I", 100)]), FbxNode("Count", [("I", n_objs)])])
    return [header, gs, defs, objects, conns]

def _axis_score(lo, hi, ext, tol):
    p = 2.0 / ext
    tol = max(tol * p, 1.5 / 32767.0)
    cross = np.floor((hi + 1.0 - tol) / p) > np.floor((lo + 1.0 + tol) / p)
    tile = np.floor(((lo + hi) * 0.5 + 1.0) / p).astype(np.int64)
    fill = 0.0
    for t in np.unique(tile):
        m = tile == t
        fill += m.sum() * (min((hi[m].max() + 1.0 - t * p) / p, 1.0) - max((lo[m].min() + 1.0 - t * p) / p, 0.0))
    return fill / len(lo), float(np.mean(cross))

def fit_uv_axis(lo, hi, max_ext=48.0):
    exts = np.arange(0.5, max_ext, 0.02)
    coarse = np.array([(lambda f, s: f - 30 * s)(*_axis_score(lo, hi, e, 0.04)) for e in exts])
    peaks = [i for i in range(len(exts)) if coarse[i] > 0.3 and
             coarse[i] >= coarse[max(0, i - 3):i + 4].max() - 1e-12]
    peaks = sorted(peaks, key=lambda i: -coarse[i])[:8]
    best = None
    for i in peaks:
        for e in np.arange(exts[i] - 0.03, exts[i] + 0.0301, 0.0005):
            if e <= 0.3: continue
            f, s = _axis_score(lo, hi, e, 0.001)
            sc = f - 30 * s
            if best is None or sc > best[0]: best = (sc, float(e), f, s)
    if best is None or best[3] > 0.002: return None
    ext = best[1]
    for q in (0.5, 0.1, 0.05):
        r = round(ext / q) * q
        if abs(r - ext) <= 0.006 * ext:
            f, s = _axis_score(lo, hi, r, 0.001)
            if s <= 0.002 and f >= best[2] - 0.02: ext = r; break
    f, s = _axis_score(lo, hi, ext, 0.001)
    return ext, f, s

def _tri_gradient_ratio(pos, uv, idx):
    a, b, c = idx.T
    e1, e2 = pos[b] - pos[a], pos[c] - pos[a]
    d1, d2 = uv[b] - uv[a], uv[c] - uv[a]
    area = np.linalg.norm(np.cross(e1, e2), axis=1)
    g11, g12, g22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    det = g11 * g22 - g12 ** 2
    with np.errstate(all="ignore"):
        def grad(s1, s2):
            a1, a2 = (g22 * s1 - g12 * s2) / det, (-g12 * s1 + g11 * s2) / det
            return np.linalg.norm(e1 * a1[:, None] + e2 * a2[:, None], axis=1)
        gu, gv = grad(d1[:, 0], d2[:, 0]), grad(d1[:, 1], d2[:, 1])
        ok = (area > 1e-6) & np.isfinite(gu) & np.isfinite(gv) & (gu > 0) & (gv > 0)
        return np.log2(gu[ok] / gv[ok]), area[ok]

def _snap(e):
    for q in (0.5, 0.1):
        r = round(e / q) * q
        if r > 0 and abs(r - e) <= 0.003 * e: return r
    return e

def detect_uv_grid(loaded):
    tris = np.concatenate([t[2][t[0]] for t in loaded.values()], axis=0)
    lo_u, hi_u = tris[:, :, 0].min(1), tris[:, :, 0].max(1)
    lo_v, hi_v = tris[:, :, 1].min(1), tris[:, :, 1].max(1)
    lr, w = [], []
    for idx, pos, uv, *_ in loaded.values():
        x, y = _tri_gradient_ratio(pos, uv, idx); lr.append(x); w.append(y)
    lr, w = np.concatenate(lr), np.concatenate(w)
    ratio = None
    if len(lr):
        o = np.argsort(lr); cw = np.cumsum(w[o]) / w.sum(); ratio = float(2 ** lr[o][np.searchsorted(cw, 0.5)])
    fu, fv = fit_uv_axis(lo_u, hi_u), fit_uv_axis(lo_v, hi_v)
    def mismatch(eu, ev): return abs(ev / eu / ratio - 1) if ratio else 0.0
    if fu is not None and fv is not None and mismatch(fu[0], fv[0]) < 0.01:
        return dict(ok=True, ext=(fu[0], fv[0]), ratio=ratio, fill=(fu[1], fv[1]), consistent=True)
    cands = []
    if ratio:
        if fv is not None:
            e = fv[0] / ratio; f, st = _axis_score(lo_u, hi_u, e, 0.001)
            if st <= 0.15 and f >= 0.9: cands.append((f + fv[1] - 3 * st, (e, fv[0]), (f, fv[1])))
        if fu is not None:
            e = fu[0] * ratio; f, st = _axis_score(lo_v, hi_v, e, 0.001)
            if st <= 0.15 and f >= 0.9: cands.append((f + fu[1] - 3 * st, (fu[0], e), (fu[1], f)))
    if cands:
        _, ext, fill = max(cands, key=lambda c: c[0])
        ext = tuple(_snap(x) for x in ext)
        return dict(ok=True, ext=ext, ratio=ratio, fill=fill, consistent=True)
    if fu is not None and fv is not None:
        return dict(ok=True, ext=(fu[0], fv[0]), ratio=ratio, fill=(fu[1], fv[1]), consistent=False)
    return dict(ok=False, ratio=ratio, fu=fu, fv=fv)

UV_PAD = ((0.0076, 0.0082), (0.0076, 0.0065))

def uv_box(ext, pad=None):
    pad = UV_PAD if pad is None else pad
    box = []
    for a in range(2):
        hi = ext[a] - pad[a][1] if abs(ext[a] - round(ext[a])) < 0.02 else ext[a]
        box.append((pad[a][0], hi))
    return box

def unsquash_uv(uv, idx, ext, mode="tile", flip_v=True, eps=1e-3, box=None):
    bx = box if box is not None else uv_box(ext)
    t = np.stack([bx[a][0] + (uv[:, a] + 1.0) * 0.5 * (bx[a][1] - bx[a][0]) for a in range(2)], 1)
    tile = np.floor(t + eps)
    tri_tile = np.floor(t[idx].mean(axis=1))
    for c in range(3): tile[idx[:, c]] = tri_tile
    local = t - tile
    rows = int(np.ceil(ext[1] - eps))
    if flip_v: local[:, 1] = 1.0 - local[:, 1]
    if mode == "udim":
        local[:, 0] += tile[:, 0]
        local[:, 1] += (rows - 1 - tile[:, 1]) if flip_v else tile[:, 1]
    return local, tile.astype(np.int64)

VRM_SIZE = 320
VRM_QUANT_SCALE, VRM_QUANT_OFFSET = 184, 196
VRM_UV_SCALE, VRM_UV_OFFSET = 208, 216

def read_exact_uv_quant(v, meshes=None):
    meshes = meshes if meshes is not None else parse_meshes(v)
    if not meshes: raise ValueError("no mesh to anchor _vrender_model_t")
    center, half = global_box(meshes)
    key = np.concatenate([half, center]).astype(np.float32)
    tol = np.maximum(np.abs(key) * 1e-3, 1e-4 * max(float(np.abs(half).max()), 1e-6))
    d, found = v.data, []
    for sh in range(4):
        n = (len(d) - sh) // 4
        if n < 10: continue
        f = np.frombuffer(d, dtype="<f4", count=n, offset=sh)
        L = n - 9
        with np.errstate(all="ignore"):
            js = np.flatnonzero(np.abs(f[:L] - key[0]) <= tol[0])
            for i in range(1, 6):
                if not js.size: break
                js = js[np.abs(f[js + i] - key[i]) <= tol[i]]
            for j in js:
                sc = f[j + 6:j + 8].astype(np.float64); of = f[j + 8:j + 10].astype(np.float64)
                if np.isfinite(sc).all() and np.isfinite(of).all() and (sc > 0).all() and (sc < 1e3).all():
                    found.append((sh + 4 * int(j) - VRM_QUANT_SCALE, sc, of))
    if not found: raise ValueError("_vrender_model_t not found in file")
    if any(not (np.allclose(c[1], found[0][1]) and np.allclose(c[2], found[0][2])) for c in found):
        print(f"warning: {len(found)} _vrender_model_t candidates with different UVs; using the first "
              f"(@0x{found[0][0]:x})", file=sys.stderr)
    return found[0][1], found[0][2]

def collect_items(v, lods=(0,), all_lods=False, rig=True, uv_mode="auto", flip_v=True, normal_refs=None, uv_grid=None, exact=None, compare=False):
    meshes = parse_meshes(v)
    if not meshes: raise RuntimeError("no mesh records found")
    if len(meshes) != v.n_buffers:
        print(f"warning: {len(meshes)} mesh records for {v.n_buffers} buffers", file=sys.stderr)
    gbox = global_box(meshes)
    group = lod_groups(meshes)
    mats, how = material_names(v)
    n_groups = max(group) + 1
    want = range(n_groups) if all_lods else [l for l in lods if l < n_groups]
    items, missing, dedup = [], [], []
    bones = parse_skeleton(v) if rig else None
    loaded, rps = {}, {}
    load_all = compare or (uv_mode != "raw" and exact is None and not uv_grid)   # heuristic UV needs every mesh
    for k, m in enumerate(meshes):
        if not load_all and group[k] not in want: continue
        rp, sub = resolve_raw(v, meshes, k)
        if rp is None: continue
        try: loaded[k] = load_mesh_data(v, k, m, gbox, rp); rps[k] = (rp, sub)
        except ValueError as e: print(f"warning: raw {k}: {e}", file=sys.stderr)

    ext, mode, box = None, "raw", None
    if uv_mode != "raw" and loaded:
        if exact is not None:
            sc, of = exact
            box = [(float(of[a] - sc[a]), float(of[a] + sc[a])) for a in range(2)]
            ext, mode = (box[0][1], box[1][1]), ("tile" if uv_mode == "auto" else uv_mode)
            print(f"note: exact UV from metadata: scale=({sc[0]:.6f},{sc[1]:.6f}) offset=({of[0]:.6f},{of[1]:.6f}) "
                  f"-> box U[{box[0][0]:.5f},{box[0][1]:.5f}] V[{box[1][0]:.5f},{box[1][1]:.5f}] tiles", file=sys.stderr)
            if uv_grid or normal_refs:
                print("note: --uv-grid / --normal-reference ignored (exact UV does not need estimation)", file=sys.stderr)
            if compare:
                try:
                    g = detect_uv_grid(loaded)
                    if g["ok"]:
                        ob = uv_box(g["ext"])
                        d = max(abs(ob[a][b] - box[a][b]) for a in range(2) for b in range(2))
                        print(f"comparison: old heuristic ext={g['ext'][0]:.4f}x{g['ext'][1]:.4f} box U{tuple(round(x,5) for x in ob[0])} "
                              f"V{tuple(round(x,5) for x in ob[1])}; max diff from exact = {d:.5f} tile (~{d*2048:.1f}px at 2048)", file=sys.stderr)
                    else:
                        print("comparison: old heuristic did not detect grid (exact worked anyway)", file=sys.stderr)
                except Exception as e:
                    print(f"comparison: heuristic failed ({e})", file=sys.stderr)
        elif uv_grid:
            ext, mode = tuple(uv_grid), ("tile" if uv_mode == "auto" else uv_mode)
            print(f"note: forced UV grid: {ext[0]:g} x {ext[1]:g} tiles", file=sys.stderr)
        else:
            g = detect_uv_grid(loaded)
            if g["ok"] and g["consistent"]:
                ext = g["ext"]; mode = "tile" if uv_mode == "auto" else uv_mode
                rt = f"; measured geometry ratio {g['ratio']:.3f} (grid {ext[1] / ext[0]:.3f})" if g["ratio"] else ""
                print(f"note: UV grid {ext[0]:.3f} x {ext[1]:.3f} tiles (stored space step "
                      f"{2 / ext[0]:.4f} x {2 / ext[1]:.4f}); fill {g['fill'][0]:.2f}/{g['fill'][1]:.2f}{rt}", file=sys.stderr)
            else:
                print("warning: UV grid not reliably detected; UV kept raw. "
                      "If you know the extension (e.g., 3,2), use --uv-grid U,V", file=sys.stderr)

    def converted_uv(k, uv):
        if ext is None: return uv, None
        return unsquash_uv(uv, loaded[k][0], ext, mode, flip_v, box=box)

    if ext is not None:
        tiles = {}
        for k, (idx, pos, uv, *_) in loaded.items():
            if group[k] not in want: continue
            _, tj = converted_uv(k, uv)
            mname = mats[meshes[k].material] if meshes[k].material < len(mats) else f"material_{meshes[k].material}"
            for ti, tjj in {tuple(t) for t in tj[np.unique(idx)]}:
                tiles.setdefault((ti, tjj), set()).add(mname)
        rows = int(np.ceil(ext[1] - 1e-3))
        for (ci, ri) in sorted(tiles, key=lambda t: (t[1], t[0])):
            rb = rows - 1 - ri if flip_v else ri
            print(f"      tile col={ci} engine_lin={ri} UDIM {1001 + ci + 10 * rb} <- {', '.join(sorted(tiles[(ci, ri)]))}", file=sys.stderr)

    uv_offsets = {}
    if ext is not None and normal_refs and box is None:
        for material_name, normal_map_path in normal_refs.items():
            keys = [k for k, m in enumerate(meshes) if k in loaded and group[k] in want
                    and (mats[m.material] if m.material < len(mats) else f"material_{m.material}") == material_name]
            if not keys: continue
            fit_data = []
            for k in keys:
                idx, pos, uv, skin, normal, tangent = loaded[k]
                fit_data.append((idx, pos, converted_uv(k, uv)[0], normal, tangent))
            offset, base_score, fit_score = _fit_uv_offset(fit_data, normal_map_path)
            uv_offsets[material_name] = offset
            print(f"note: UV offset of {material_name} measured by normal map + geometric normals: "
                  f"({offset[0]:+.5f},{offset[1]:+.5f}) = "
                  f"({offset[0]*1024:+.1f},{offset[1]*1024:+.1f}) px at 1024; "
                  f"alignment {base_score:.4f}->{fit_score:.4f}", file=sys.stderr)
    for k, m in enumerate(meshes):
        if group[k] not in want: continue
        if k not in loaded:
            if k not in rps: missing.append(k)
            continue
        rp, sub = rps[k]
        if sub: dedup.append((k, int(re.search(r"\.(\d+)\.nvn_gen2\.raw$", rp).group(1))))
        idx, pos, uv, skin, normal, tangent = loaded[k]
        mname = mats[m.material] if m.material < len(mats) else f"material_{m.material}"
        nm = f"{mname}_LOD{group[k]}"
        if any(it[0] == nm for it in items): nm += f"_{m.material}"
        uv = converted_uv(k, uv)[0]
        if mname in uv_offsets:
            uv = uv + np.asarray(uv_offsets[mname])
        binormal = _mesh_binormals(pos, idx, uv, normal, tangent)
        items.append((nm, mname, pos, idx, uv, skin if bones else None, normal, tangent, binormal))
    if dedup: print(f"note: missing .raw replaced by identical copy (dedup): {', '.join(f'{k}<-{j}' for k, j in dedup)}", file=sys.stderr)
    if missing: print(f"warning: missing .raw without equivalent (meshes skipped): {missing}", file=sys.stderr)
    if not items: raise RuntimeError("no meshes exported")
    return items, bones, n_groups, list(want), how

def convert_model(v, outdir, lods=(0,), all_lods=False, rig=True, uv_mode="auto", flip_v=True, normal_refs=None, uv_grid=None,
                  exact_uv=True, compare=False):
    exact = None
    if exact_uv and uv_mode != "raw":
        try: exact = read_exact_uv_quant(v)
        except Exception as e:
            print(f"warning: exact UV unavailable ({type(e).__name__}: {e}); using heuristic detection", file=sys.stderr)
    items, bones, n_groups, want, how = collect_items(v, lods, all_lods, rig, uv_mode, flip_v, normal_refs, uv_grid, exact, compare)
    src = os.path.basename(v.source_path) if v.source_path else v.base
    out = os.path.join(outdir, os.path.splitext(src)[0] + ".fbx")
    os.makedirs(outdir, exist_ok=True)
    write_fbx(out, build_fbx(items, bones))
    tv = sum(len(i[2]) for i in items); tt = sum(len(i[3]) for i in items)
    print(f"[model] {os.path.basename(v.path)} -> {out}  ({len(items)} meshes, groups/LOD {sorted(set(want))} of {n_groups}, {tv} vertices, {tt} triangles)")
    print(f"         material names: {how}")
    if bones:
        n_sk = sum(1 for it in items if it[5] is not None)
        print(f"         rig: {len(bones)} bones (root '{bones[0].name}'), {n_sk} meshes with skin")
    return out

def parse_bullet_mesh(v):
    d = v.data
    i = d.index(b"BULLET"); p = i + 12
    arrays = []
    while p + 24 <= len(d):
        tag = d[p:p + 4]
        if not re.fullmatch(rb"[A-Z0-9 ]{4}", tag): break
        ln, = struct.unpack_from("<i", d, p + 4); nr, = struct.unpack_from("<i", d, p + 20)
        if tag == b"ARAY" and nr > 0 and ln == nr * (ln // nr): arrays.append((nr, ln // nr, d[p + 24:p + 24 + ln]))
        p += 24 + ln
    verts = [a for a in arrays if a[1] == 16 and np.isfinite(np.frombuffer(a[2], "<f4")).all()
             and not np.frombuffer(a[2], "<f4").reshape(-1, 4)[:, 3].any()]
    tris = [a for a in arrays if a[1] == 8]
    for nv, _, vb in verts:
        pos = np.frombuffer(vb, "<f4").reshape(-1, 4)[:, :3].astype(np.float64)
        for nt, _, tb in tris:
            t = np.frombuffer(tb, "<u2").reshape(-1, 4)
            if not t[:, 3].any() and t[:, :3].max() < nv: return pos, t[:, :3].astype(np.int32)
    raise RuntimeError("collision mesh not found in Bullet blob")

def convert_collision(v, outdir):
    pos, idx = parse_bullet_mesh(v)
    src = os.path.basename(v.source_path) if v.source_path else v.base
    out = os.path.join(outdir, os.path.splitext(src)[0] + ".fbx")
    os.makedirs(outdir, exist_ok=True)
    write_fbx(out, build_fbx([(os.path.splitext(src)[0], "collision", pos, idx, None)]))
    print(f"[collision] {os.path.basename(v.path)} -> {out}  ({len(pos)} vertices, {len(idx)} triangles)")
    return out

def info(v, exact_uv=True):
    print(f"{v.path}\n  version={v.version} buffers(.raw)={v.n_buffers} type={'texture' if v.is_texture else 'model' if v.is_model else 'collision (Bullet)' if v.is_bullet else '?'}")
    print(f"  source: {v.source_path}")
    if v.is_texture:
        for fmt, w, h, mips, pos in find_texture_descs(v.data):
            print(f"  descriptor @0x{pos:x}: VkFormat {fmt} ({VKFMT[fmt][0]}) {w}x{h} mips={mips}")
    elif v.is_model:
        ms = parse_meshes(v)
        mats, how = material_names(v)
        print(f"  meshes: {len(ms)}  groups/LOD: {max(lod_groups(ms)) + 1 if ms else 0}  materials ({how}): {mats}")
        if exact_uv:
            try:
                sc, of = read_exact_uv_quant(v, ms)
                print(f"  exact UV: scale=({sc[0]:.6f},{sc[1]:.6f}) offset=({of[0]:.6f},{of[1]:.6f}) "
                      f"-> U[{of[0]-sc[0]:.5f},{of[0]+sc[0]:.5f}] V[{of[1]-sc[1]:.5f},{of[1]+sc[1]:.5f}] tiles")
            except Exception as e: print(f"  exact UV unavailable: {type(e).__name__}: {e}")
        sk = parse_skeleton(v)
        if sk: print(f"  skeleton: {len(sk)} bones, root '{sk[0].name}', e.g.: {[b.name for b in sk[1:5]]}")
        for k, m in enumerate(ms[:6]):
            print(f"   [{k}] mat={m.material} V={m.V} I={m.I} idx={'u32' if m.idx32 else 'u16'} skin={m.skinned} streams@{[hex(o) for o in m.offs]}")

def _worker_init(uv_pad, png_level):
    global UV_PAD, PNG_LEVEL
    if uv_pad is not None: UV_PAD = ((uv_pad, uv_pad), (uv_pad, uv_pad))
    PNG_LEVEL = png_level

def _process_one(task):
    """Converts one file; returns (stdout, stderr) text so the parent prints logs in order."""
    import io, contextlib
    f, outdir, o = task
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            v = Vipr(f)
            if v.is_texture: convert_texture(v, outdir, o["mip"])
            elif v.is_model: convert_model(v, outdir, (o["lod"],), o["all_lods"], o["rig"], o["uv"], o["flip_v"],
                                           o["normal_refs"], o["uv_grid"], o["exact_uv"], o["compare"])
            elif v.is_bullet: convert_collision(v, outdir)
            else: print(f"[skipped] {f}: unknown type")
        except Exception as e:
            print(f"[error] {f}: {e}", file=sys.stderr)
    return out.getvalue(), err.getvalue()

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="+", help=".import.vipr file or folder (or: info <file>)")
    ap.add_argument("-o", "--out", default="Extracted_out", help="output directory; input subfolder structure is preserved inside it")
    ap.add_argument("--lod", type=int, default=0, help="model LOD (default 0)")
    ap.add_argument("--all-lods", action="store_true")
    ap.add_argument("--mip", type=int, default=0, help="texture mip level (default 0)")
    ap.add_argument("--uv", choices=["auto", "tile", "udim", "raw"], default="auto",
                    help="UVs: tile = un-flatten and take each vertex to tile 0..1 (one texture per material, as the game samples); "
                         "udim = same, but keeps the tile number (1001+col+10*row); raw = raw -1..1 values; "
                         "auto (default) = tile if model UV grid is detected, else raw")
    ap.add_argument("--uv-pad", metavar="P", type=float, help="content margin at UV box extremes, in tiles (default calibrated ~0.0076; 0 = exact 0..N box)")
    ap.add_argument("--uv-grid", metavar="U,V", help="forces model UV grid extension in tiles (e.g., 3,2 or 6,3.1); only use if auto-detection fails")
    ap.add_argument("--no-flip-v", action="store_true", help="do not flip V (use if textures appear upside down)")
    ap.add_argument("--normal-reference", action="append", default=[], metavar="MATERIAL=PNG",
                    help="normal map to adjust only the UV offset using geometric normals; can be repeated")
    ap.add_argument("-j", "--jobs", type=int, default=0, help="parallel processes (default: number of CPUs; 1 = sequential)")
    ap.add_argument("--png-level", type=int, default=1, choices=range(0, 10), metavar="0-9",
                    help="PNG compression (default 1 = fast; 9 = smallest, slower; always lossless)")
    ap.add_argument("--no-rig", action="store_true", help="do not export skeleton/skin")
    ap.add_argument("--no-exact-uv", action="store_true", help="do not read exact UV from model metadata; uses heuristic detection only")
    ap.add_argument("--compare-uv", action="store_true", help="with exact UV, prints the difference from the old heuristic")
    a = ap.parse_args()
    if a.uv_pad is not None:
        global UV_PAD
        UV_PAD = ((a.uv_pad, a.uv_pad), (a.uv_pad, a.uv_pad))
    global PNG_LEVEL
    PNG_LEVEL = a.png_level
    uv_grid = None
    if a.uv_grid:
        try: uv_grid = tuple(float(x) for x in a.uv_grid.split(","))
        except ValueError: uv_grid = None
        if not uv_grid or len(uv_grid) != 2 or min(uv_grid) <= 0: ap.error("--uv-grid expects two positive numbers: U,V")
    normal_refs = {}
    for spec in a.normal_reference:
        try:
            name, path = spec.split("=", 1)
            if not name or not os.path.isfile(path): raise ValueError("empty material or PNG not found")
            normal_refs[name] = path
        except Exception as e:
            ap.error(f"invalid --normal-reference ({spec!r}): {e}")
    exact_uv = not a.no_exact_uv
    if a.path[0] == "info":
        for p in a.path[1:]: info(Vipr(p), exact_uv)
        return
    files = []
    for p in a.path:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                rel = os.path.relpath(r, p)
                files += [(os.path.join(r, f), "" if rel == "." else rel) for f in fs if f.endswith(".vipr")]
        else: files.append((p, ""))
    opts = dict(mip=a.mip, lod=a.lod, all_lods=a.all_lods, rig=not a.no_rig, uv=a.uv, flip_v=not a.no_flip_v,
                normal_refs=normal_refs, uv_grid=uv_grid, exact_uv=exact_uv, compare=a.compare_uv)
    tasks = []
    for f, rel in files:
        try: size = os.path.getsize(f)
        except OSError: size = 0
        tasks.append((size, f, os.path.join(a.out, rel) if rel else a.out))
    tasks.sort(key=lambda t: (-t[0], t[1]))          # biggest first = better load balance
    tasks = [(f, o, opts) for _, f, o in tasks]
    jobs = a.jobs if a.jobs and a.jobs > 0 else (os.cpu_count() or 1)
    jobs = min(jobs, len(tasks))
    init = (None if a.uv_pad is None else a.uv_pad, a.png_level)
    done = 0
    if jobs > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init, initargs=init) as ex:
                for out, err in ex.map(_process_one, tasks, chunksize=1):
                    sys.stdout.write(out); sys.stderr.write(err); sys.stdout.flush(); done += 1
            return
        except Exception as e:
            print(f"warning: parallel mode failed ({type(e).__name__}: {e}); continuing sequentially", file=sys.stderr)
    _worker_init(*init)
    for t in tasks[done:]:
        out, err = _process_one(t)
        sys.stdout.write(out); sys.stderr.write(err); sys.stdout.flush()

if __name__ == "__main__":
    main()