# VZIP / VIPR Tools

Python tools to extract the files from a `data.zip` (Eclipse engine, Star Fox) and convert `.vipr` assets back to common formats (FBX, PNG, TGA).

## Contents

| File | Description |
|---|---|
| `vzip_extract_named.py` | Extracts a `data.zip` recovering the real file names |
| `vzip_Extract_DragAndDrop.bat` | Windows shortcut: drag the `data.zip`/`.vzip` onto the BAT |
| `vipr_model_textures.py` | Converts `.import.vipr` + `.nvn_gen2.raw` into models and textures |
| `Model_Textures_Extract.bat` | Windows shortcut: converts all `.vipr` files in the folder and subfolders |

## Requirements

- Python 3
- `lz4` (VZIP): `pip install lz4`
- `numpy`, `pillow`, `texture2ddecoder` (VIPR): `pip install numpy pillow texture2ddecoder`

The `.bat` files check for and install the modules automatically.

---

## 1. VZIP Extraction

### Windows (drag and drop)

Drag the `data.zip` / `.vzip` file onto `vzip_Extract_DragAndDrop.bat`.
The output is created next to the file, in `<name>_extracted/`.

### Command line

```
python vzip_extract_named.py <file.vzip> [-o FOLDER] [options]
```

| Option | Description |
|---|---|
| `-o`, `--out DIR` | Output folder (default: `output`) |
| `--dry-run` | Only builds `names_map.csv` and prints a summary, extracts nothing |
| `--limit N` | Extracts only the first N members (quick test) |
| `--skip-audio` | Does not extract unnamed audio (RIFF/BKHD) |
| `--only-named` | Extracts only files whose real name was recovered |
| `--zip-only` | Ignores table records that have no ZIP member |

Examples:

```
python vzip_extract_named.py data.zip --dry-run
python vzip_extract_named.py data.zip -o output --limit 500
python vzip_extract_named.py data.zip -o output --skip-audio
```

> A full extraction can take up about 33 GB.

### How it works

- The file is a ZIP with a `VZIP` preamble table. Each member's data is a chain of LZ4 blocks (`[u32 size][LZ4 block]`), each using the previous 64 KB as its dictionary.
- Each member name is the 64-bit FNV-1a hash of the path without the leading slash.
- Paths are recovered from the `/data/...` strings inside the `vjson-1` JSON members and from the source path embedded in VIPR binaries. Every match is validated by the hash.
- Members with no recoverable path go to `_unnamed/` (mostly Wwise audio) or `_meta/` (package list JSONs).

---

## 2. VIPR Conversion

Converts the extracted assets to common formats:

| Input | Output |
|---|---|
| `<name>.<png\|tga>.import.vipr` + `.N.nvn_gen2.raw` | `<name>.<png\|tga>` |
| `<name>.fbx.import.vipr` + `.N.nvn_gen2.raw` | `<name>.fbx` (binary 7.4, with skeleton and skin) |
| `<name>.collision.fbx.import.vipr` | `<name>.collision.fbx` |

Supported texture formats: BC1, BC2, BC3, BC4, BC5, BC7 and ASTC 4x4.

The `.raw` files must be in the same folder as the matching `.import.vipr`.

### Windows

Place `Model_Textures_Extract.bat` and `vipr_model_textures.py` in the folder containing the `.vipr` files and run the BAT.
It processes the folder and subfolders with `--all-lods`, and saves everything to `Extracted_out/`, preserving the folder structure.

### Command line

```
python vipr_model_textures.py <file.import.vipr | folder> [-o OUTPUT] [options]
python vipr_model_textures.py info <file.import.vipr>
```

| Option | Description |
|---|---|
| `-o`, `--out DIR` | Output folder (default: `Extracted_out`) |
| `--lod N` / `--all-lods` | Model LOD (default: 0) or all LODs |
| `--mip N` | Texture mip level (default: 0) |
| `--no-rig` | Does not export skeleton/skin |
| `--uv auto\|tile\|udim\|raw` | UV mode (default: `auto`) |
| `--uv-grid U,V` | Forces the UV grid extent, only if auto-detection fails |
| `--uv-pad P` | Content margin at the UV extremes, in tiles |
| `--no-flip-v` | Does not flip V (use if textures appear upside down) |
| `--no-exact-uv` | Does not read exact UVs from metadata, uses the heuristic only |
| `--compare-uv` | Prints the difference between exact UVs and the heuristic |
| `--normal-reference MATERIAL=PNG` | Normal map reference per material |
| `-j`, `--jobs N` | Parallel processes (default: number of CPUs; 1 = sequential) |
| `--png-level 0-9` | PNG compression level (default: 1) |

---

## Workflow

1. Extract the `data.zip` with `vzip_extract_named.py`.
2. Run `Model_Textures_Extract.bat` (or `vipr_model_textures.py`) on the extracted folder to get the models and textures.

---

## Notice

This project was developed with the assistance of AI. The code and documentation may contain mistakes, so please review the results before relying on them.

## Contributing

Anyone is free to contribute fixes and improvements. Issues and pull requests are welcome.

## Disclaimer

This is an unofficial, community reverse-engineering project. It is not affiliated with or endorsed by the owners of the games or engine it works with. It contains no game assets, and you must supply your own legally obtained files. The software is provided "as is", without warranty of any kind.

## License

Released under the [GNU General Public License v3.0](LICENSE). You are free to use, modify and share this project, but any distributed derivative work must also be released under GPL-3.0 with its source code available.
