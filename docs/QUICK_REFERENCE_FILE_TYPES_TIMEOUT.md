# Quick Reference: File Types & Timeout Configuration

## 📄 Supported File Types (200+)

### Most Common Types

| Type | Extensions | Auto-loads as |
|------|-----------|---------------|
| **CSV** | `.csv`, `.tsv`, `.tab` | pandas DataFrame |
| **Excel** | `.xlsx`, `.xls`, `.xlsm` | pandas DataFrame |
| **JSON** | `.json`, `.jsonl` | DataFrame or dict |
| **Parquet** | `.parquet` | pandas DataFrame |
| **Pickle** | `.pkl`, `.pickle` | Python object |
| **NumPy** | `.npy`, `.npz` | NumPy array |
| **HDF5** | `.h5`, `.hdf5` | pandas DataFrame |
| **SQLite** | `.db`, `.sqlite` | sqlite3.Connection |
| **Text** | `.txt`, `.log`, `.md` | String |
| **YAML** | `.yaml`, `.yml` | dict |
| **Image** | `.png`, `.jpg`, `.jpeg` | File path (for PIL) |
| **Audio** | `.mp3`, `.wav`, `.flac` | File path (for librosa) |

## ⚙️ Configuration (.env)

```bash
# Code execution timeout (seconds) - Only counts actual code runtime
CODE_EXECUTION_TIMEOUT=300  # Default: 5 minutes

# File limits
FILE_EXPIRATION_HOURS=48    # Files expire after 48 hours
MAX_FILES_PER_USER=20       # Max files per user
```

## 💻 Usage Examples

### Load Data Files
```python
# CSV
df = load_file('file_id')  # → pd.read_csv()

# Excel
df = load_file('file_id')  # → pd.read_excel()

# Parquet
df = load_file('file_id')  # → pd.read_parquet()

# JSON
data = load_file('file_id')  # → pd.read_json() or json.load()
```

### Load Config Files
```python
# YAML
config = load_file('file_id')  # → yaml.safe_load()

# TOML  
config = load_file('file_id')  # → toml.load()

# JSON
config = load_file('file_id')  # → json.load()
```

### Load Binary/Scientific
```python
# NumPy
array = load_file('file_id')  # → np.load()

# Pickle
obj = load_file('file_id')  # → pd.read_pickle()

# HDF5
df = load_file('file_id')  # → pd.read_hdf()

# Stata
df = load_file('file_id')  # → pd.read_stata()
```

### Load Media Files
```python
# Images (returns path for PIL/OpenCV)
img_path = load_file('file_id')
from PIL import Image
img = Image.open(img_path)

# Audio (returns path for librosa)
audio_path = load_file('file_id')
import librosa
y, sr = librosa.load(audio_path)

# Video (returns path for moviepy)
video_path = load_file('file_id')
from moviepy.editor import VideoFileClip
clip = VideoFileClip(video_path)
```

## ⏱️ Timeout Behavior

```
┌──────────────────────────────┐
│  NOT counted in timeout:     │
├──────────────────────────────┤
│  • File upload               │
│  • Venv setup                │
│  • Package installation      │
│  • Code validation           │
└──────────────────────────────┘

┌──────────────────────────────┐
│  ⏱️  COUNTED in timeout:     │
├──────────────────────────────┤
│  • Python code execution     │
│  • Data processing           │
│  • Model training            │
│  • File generation           │
└──────────────────────────────┘

┌──────────────────────────────┐
│  NOT counted in timeout:     │
├──────────────────────────────┤
│  • Result collection         │
│  • File upload to Discord    │
└──────────────────────────────┘
```

## 🎯 Recommended Timeouts

| Use Case | Timeout | Command |
|----------|---------|---------|
| Quick analysis | 60s | `CODE_EXECUTION_TIMEOUT=60` |
| Normal (default) | 300s | `CODE_EXECUTION_TIMEOUT=300` |
| ML training | 900s | `CODE_EXECUTION_TIMEOUT=900` |
| Heavy processing | 1800s | `CODE_EXECUTION_TIMEOUT=1800` |

## 📊 Complete File Type List

### Data Formats (40+)
CSV, TSV, Excel (XLSX/XLS), ODS, JSON, JSONL, XML, YAML, TOML, Parquet, Feather, Arrow, HDF5, Pickle, NumPy (NPY/NPZ), MATLAB (MAT), SPSS (SAV), Stata (DTA), SAS, R Data, Avro, ORC, Protobuf, MessagePack, BSON, SQLite, SQL

### Images (20+)
PNG, JPEG, GIF, BMP, TIFF, WebP, SVG, ICO, HEIC, RAW, CR2, NEF, DNG, PSD, AI, EPS

### Audio (10+)
MP3, WAV, FLAC, AAC, OGG, M4A, WMA, OPUS, AIFF, APE

### Video (15+)
MP4, AVI, MKV, MOV, WMV, FLV, WebM, M4V, MPG, MPEG, 3GP

### Documents (10+)
PDF, DOC/DOCX, ODT, RTF, TXT, Markdown, LaTeX, EPUB, MOBI

### Programming (50+)
Python, R, JavaScript, TypeScript, Java, C/C++, C#, Go, Rust, Ruby, PHP, Swift, Kotlin, Scala, Shell, PowerShell, Lua, Julia, and 30+ more

### Archives (15+)
ZIP, TAR, GZ, BZ2, XZ, 7Z, RAR, TGZ, TBZ, LZMA, ZST

### Geospatial (10+)
GeoJSON, Shapefile, KML, KMZ, GPX, GML, Geodatabase

### Scientific (15+)
FITS, DICOM, NIfTI, VTK, STL, OBJ, PLY, FBX, GLTF

### Configuration (10+)
INI, CFG, CONF, Properties, ENV, YAML, TOML, XML, JSON

## 🚨 Error Handling

### Timeout Error
```python
# If execution exceeds timeout:
TimeoutError: Code execution exceeded 300 seconds
```

### File Not Found
```python
# If file_id doesn't exist:
ValueError: File abc123 not found or not accessible
```

### Unsupported Operation
```python
# If file type doesn't support requested operation:
# AI will generate appropriate error handling code
```

## 💡 Tips

1. **Large Files**: Increase timeout for processing large datasets
2. **ML Training**: Set timeout to 15-30 minutes for model training
3. **Images**: Use PIL/OpenCV after loading path
4. **Audio/Video**: Use specialized libraries (librosa, moviepy)
5. **Multiple Files**: Load multiple files in same execution
6. **Archives**: Extract archives programmatically in Python

## 📚 Related Documentation

- `UNIFIED_FILE_SYSTEM_SUMMARY.md` - Complete file system overview
- `ALL_FILE_TYPES_AND_TIMEOUT_UPDATE.md` - Detailed implementation guide
- `CODE_INTERPRETER_GUIDE.md` - Code execution details
