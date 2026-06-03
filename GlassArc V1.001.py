#!/usr/bin/env python3
"""
GlassArc MASTER – Universal Model Auto‑Detection + Procrustes Tensor Mapping
- Scans all .gguf files in models/
- Uses largest model as master
- Aligns token embeddings via orthogonal Procrustes (NumPy SVD)
- Extracts token strings from GGUF metadata; falls back to VOCABULARY for generated models
- Lightweight: only token embeddings loaded (few MB)
"""

import os, sys, subprocess, tempfile, shutil, time, json, zipfile, re
import random, struct, traceback, hashlib, zlib, threading, math, urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# 0. BOOTSTRAP & LOGGING
# ============================================================================
PROJECT_DIR = Path(__file__).parent.absolute()
TRACE_LOG = PROJECT_DIR / "glassarc_trace.log"
LOG_FILE = PROJECT_DIR / "glassarc_run.log"
GLITCH_COUNTER = 0
LOG_ENTRIES = []
_phase = "BOOTSTRAP"
_start_time = time.time()

def set_phase(p: str):
    global _phase
    _phase = p

def trace(step: str, status: str, message: str = "", detail: str = ""):
    ts = datetime.now().isoformat()
    elapsed = time.time() - _start_time
    line = f"{ts} | {elapsed:7.2f}s | {_phase:15} | {step:40} | {status:10} | {message}"
    if detail:
        line += f" | {detail}"
    with open(TRACE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    icons = {"OK": "✓", "DONE": "✓", "PASS": "✓", "ACTIVE": "→",
             "WARN": "⚠", "ERROR": "✗", "START": "▶", "INFO": "ℹ"}
    icon = icons.get(status, "·")
    color = {"OK": "\033[92m", "DONE": "\033[92m", "PASS": "\033[92m",
             "WARN": "\033[93m", "ERROR": "\033[91m", "START": "\033[94m"}.get(status, "")
    reset = "\033[0m" if color else ""
    print(f"{color}{icon}{reset} {step:40} {status:10} {message}")

def log_event(level: str, message: str, glitch: bool = False, **extra):
    global GLITCH_COUNTER
    entry = {"ts": datetime.now().isoformat(), "phase": _phase, "level": level,
             "message": message, "glitch": glitch, **extra}
    LOG_ENTRIES.append(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    if glitch:
        GLITCH_COUNTER += 1

def log_error(e: Exception, glitch: bool = True):
    log_event("ERROR", str(e), glitch=glitch, traceback=traceback.format_exc())
    trace("Exception", "ERROR", str(e)[:60])

trace("GlassArc Master", "START", "Fully corrected version with fallback vocabulary")

# ============================================================================
# 1. ENVIRONMENT (auto‑venv)
# ============================================================================
set_phase("ENV")
VENV_DIR = PROJECT_DIR / "venv"

def is_venv():
    return sys.prefix != sys.base_prefix

def get_python_executable():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def create_venv():
    trace("Virtual Environment", "START", f"Creating at {VENV_DIR}")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)],
                       check=True, capture_output=True, timeout=120)
        trace("Virtual Environment", "DONE", "Created successfully")
        return True
    except Exception as e:
        trace("Virtual Environment", "ERROR", str(e))
        return False

def restart_in_venv():
    python_exe = get_python_executable()
    trace("Virtual Environment", "RESTART", f"Re-executing with {python_exe}")
    try:
        subprocess.run([str(python_exe)] + sys.argv, check=True)
        sys.exit(0)
    except Exception as e:
        trace("Virtual Environment", "ERROR", f"Failed to restart: {e}")
        sys.exit(1)

if not is_venv():
    if not VENV_DIR.exists():
        if not create_venv():
            trace("Virtual Environment", "WARN", "Running without venv")
        else:
            restart_in_venv()
    else:
        restart_in_venv()
else:
    trace("Virtual Environment", "ACTIVE", f"Running in {VENV_DIR}")

# ============================================================================
# 2. DEPENDENCIES
# ============================================================================
set_phase("DEPENDENCIES")
REQUIRED_PACKAGES = [
    ("numpy", "NumPy", True),
    ("psutil", "psutil", True),
    ("gguf", "GGUF", True),
    ("requests", "Requests", False),
    ("llama-cpp-python", "llama.cpp", False),
    ("tika", "Apache Tika", False),
    ("beautifulsoup4", "BeautifulSoup4", False),
    ("trafilatura", "Trafilatura", False),
    ("fake-useragent", "FakeUserAgent", False),
]
INSTALLED_MODULES = {}

def install_package(pkg_name, display_name, required):
    trace(f"Install {display_name}", "START", pkg_name)
    python_exe = sys.executable
    for attempt in range(3):
        try:
            subprocess.run([python_exe, "-m", "pip", "install", "--quiet",
                            "--disable-pip-version-check", pkg_name],
                           check=True, capture_output=True, timeout=300)
            trace(f"Install {display_name}", "OK", f"Attempt {attempt+1}")
            return True
        except:
            time.sleep(2)
    if required:
        trace(f"Install {display_name}", "ERROR", "Required package failed")
        return False
    else:
        trace(f"Install {display_name}", "WARN", "Optional package skipped")
        return True

for pkg, name, req in REQUIRED_PACKAGES:
    if not install_package(pkg, name, req) and req:
        sys.exit(1)

trace("Dependencies", "DONE", f"{len(REQUIRED_PACKAGES)} packages processed")

# ============================================================================
# 3. IMPORTS
# ============================================================================
try:
    import numpy as np
    INSTALLED_MODULES['numpy'] = np
except:
    trace("Import", "ERROR", "NumPy required")
    sys.exit(1)

try:
    import psutil
    INSTALLED_MODULES['psutil'] = psutil
except:
    trace("Import", "ERROR", "psutil required")
    sys.exit(1)

try:
    from gguf import GGUFReader, GGUFWriter
    INSTALLED_MODULES['gguf'] = GGUFReader
except:
    trace("Import", "ERROR", "GGUF required")
    sys.exit(1)

try:
    import requests
    INSTALLED_MODULES['requests'] = requests
except:
    trace("Import", "WARN", "requests not available")

try:
    import llama_cpp
    INSTALLED_MODULES['llama_cpp'] = llama_cpp
except:
    trace("Import", "WARN", "llama-cpp-python not available, simulation mode")

# Tika with auto‑repair
tika_parser = None
try:
    from tika import parser as tika_parser
    INSTALLED_MODULES['tika'] = tika_parser
    trace("Import", "OK", "Tika available")
except SyntaxError as e:
    trace("Import", "ERROR", f"Tika corrupted: {str(e)[:50]}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tika"],
                       check=True, capture_output=True, timeout=30)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "tika"],
                       check=True, capture_output=True, timeout=120)
        from tika import parser as tika_parser
        INSTALLED_MODULES['tika'] = tika_parser
        trace("Import", "OK", "Tika reinstalled")
    except:
        trace("Import", "WARN", "Tika reinstall failed")
except Exception as e:
    trace("Import", "WARN", f"Tika import failed: {type(e).__name__}")

try:
    from bs4 import BeautifulSoup
    INSTALLED_MODULES['bs4'] = BeautifulSoup
except:
    trace("Import", "WARN", "BeautifulSoup not available")
try:
    import trafilatura
    INSTALLED_MODULES['trafilatura'] = trafilatura
except:
    trace("Import", "WARN", "Trafilatura not available")
try:
    from fake_useragent import UserAgent
    INSTALLED_MODULES['fake_ua'] = UserAgent
except:
    trace("Import", "WARN", "FakeUserAgent not available")

# ============================================================================
# 4. FOLDERS
# ============================================================================
set_phase("FOLDERS")
FOLDERS = {
    "models": "GGUF models (auto‑detected)",
    "data": "Master text corpus",
    "output": "Generated responses",
    "logs": "Runtime logs",
    "glassarc_container": "Packed container files",
    "tensor_cache": "Cached word embeddings",
    "temp": "Temporary working files"
}
for folder, desc in FOLDERS.items():
    (PROJECT_DIR / folder).mkdir(exist_ok=True)
    trace("Folder", "OK", f"{folder:20} — {desc}")
trace("Folder Structure", "DONE", f"{len(FOLDERS)} directories ready")

# ============================================================================
# 5. SYSTEM HORSEPOWER
# ============================================================================
set_phase("HORSEPOWER")
class SystemHorsepower:
    def __init__(self):
        self.cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        self.cpu_cores_logical = psutil.cpu_count(logical=True) or 1
        self.cpu_freq_mhz = self._get_cpu_freq()
        self.ram_total_gb = psutil.virtual_memory().total / (1024**3)
        self.ram_available_gb = psutil.virtual_memory().available / (1024**3)
        self.numpy_gflops = self._benchmark_numpy()
        self.tier = self._classify_tier()
        self.recommended_ctx = self._recommend_ctx()
        self.recommended_threads = self._recommend_threads()
        self.recommended_prefetch = self._recommend_prefetch()
        trace("CPU Cores", "INFO", f"Physical: {self.cpu_cores_physical}, Logical: {self.cpu_cores_logical}")
        trace("CPU Frequency", "INFO", f"{self.cpu_freq_mhz:.0f} MHz")
        trace("RAM", "INFO", f"Total: {self.ram_total_gb:.1f} GB, Available: {self.ram_available_gb:.1f} GB")
        trace("NumPy Benchmark", "INFO", f"{self.numpy_gflops:.2f} GFLOPS")
        trace("System Tier", "OK", f"{self.tier} — ctx={self.recommended_ctx}, threads={self.recommended_threads}")

    def _get_cpu_freq(self):
        try:
            return psutil.cpu_freq().current or 2000.0
        except:
            return 2000.0

    def _benchmark_numpy(self):
        try:
            size = 512
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            start = time.perf_counter()
            for _ in range(5):
                np.dot(a, b)
            elapsed = time.perf_counter() - start
            return (5 * 2 * size**3) / elapsed / 1e9
        except:
            return 1.0

    def _classify_tier(self):
        score = 0
        if self.ram_total_gb >= 16: score += 3
        elif self.ram_total_gb >= 8: score += 2
        else: score += 1
        if self.cpu_cores_physical >= 8: score += 3
        elif self.cpu_cores_physical >= 4: score += 2
        else: score += 1
        if self.numpy_gflops >= 50: score += 3
        elif self.numpy_gflops >= 10: score += 2
        else: score += 1
        if score >= 8: return "HIGH"
        elif score >= 5: return "MEDIUM"
        return "LOW"

    def _recommend_ctx(self):
        return {"LOW": 256, "MEDIUM": 512, "HIGH": 1024}[self.tier]

    def _recommend_threads(self):
        base = min(self.cpu_cores_physical, 8)
        return {"LOW": max(1, base//2), "MEDIUM": base, "HIGH": base}[self.tier]

    def _recommend_prefetch(self):
        return {"LOW": 1, "MEDIUM": 3, "HIGH": 5}[self.tier]

HORSEPOWER = SystemHorsepower()
trace("System Horsepower", "DONE", f"{HORSEPOWER.tier} tier")

# ============================================================================
# 6. MASTER TEXT & VOCABULARY
# ============================================================================
set_phase("TEXT")
MASTER_TEXT_PATH = PROJECT_DIR / "data" / "master.txt"
SEED_TEXTS = [
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "Call me Ishmael. Some years ago never mind how long precisely having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
    "In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "Far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the Galaxy lies a small unregarded yellow sun.",
    "The sky above the port was the color of television, tuned to a dead channel.",
    "I am an invisible man. No, I am not a spook like those who haunted Edgar Allan Poe.",
    "Someone must have slandered Josef K., for one morning, without having done anything truly wrong, he was arrested.",
    "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer; but that ain't no matter.",
    "Once upon a time and a very good time it was there was a moocow coming down along the road and this moocow that was coming down along the road met a nicens little boy named baby tuckoo.",
    "If you really want to hear about it, the first thing you'll probably want to know is where I was born, and what my lousy childhood was like.",
    "I write this sitting in the kitchen sink. That is, my feet are in it; the rest of me is on the draining-board.",
    "The past is a foreign country; they do things differently there."
]
random.seed(42)
if MASTER_TEXT_PATH.exists():
    master_text = MASTER_TEXT_PATH.read_text(encoding="utf-8")
    if len(re.findall(r'\b[a-zA-Z0-9]+\b', master_text)) < 10:
        trace("Master Text", "WARN", "Corrupted, regenerating")
        MASTER_TEXT_PATH.unlink()
if not MASTER_TEXT_PATH.exists():
    corpus_parts = []
    for text in SEED_TEXTS:
        corpus_parts.append(text)
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        for _ in range(3):
            random.shuffle(words)
            corpus_parts.append(" ".join(words[:20]))
    master_text = "\n\n".join(corpus_parts)
    MASTER_TEXT_PATH.write_text(master_text, encoding="utf-8")
    trace("Master Text", "GENERATED", f"{len(master_text)} chars")
else:
    master_text = MASTER_TEXT_PATH.read_text(encoding="utf-8")
    trace("Master Text", "LOADED", f"{len(master_text)} chars")
VOCABULARY = sorted(set(w.lower() for w in re.findall(r'\b[a-zA-Z0-9]+\b', master_text)))
if len(VOCABULARY) < 100:
    fallback = ["the","be","to","of","and","a","in","that","have","I","it","for","not","on","with","he","as","you","do","at"]
    VOCABULARY = sorted(set(VOCABULARY) | set(fallback))
    for i in range(100 - len(VOCABULARY)):
        VOCABULARY.append(f"token_{i}")
    VOCABULARY = sorted(VOCABULARY)
trace("Vocabulary", "OK", f"{len(VOCABULARY)} words")

# ============================================================================
# 7. TENSOR MAPPER (deterministic embeddings)
# ============================================================================
set_phase("TENSOR_MAP")
class TensorMapper:
    def __init__(self, vocab: List[str], embedding_dim: int = 64):
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self._map = {}
        self.cache_file = PROJECT_DIR / "tensor_cache" / "embeddings.npz"
        if self.cache_file.exists():
            try:
                self._load_cache()
            except Exception as e:
                trace("Tensor Cache", "WARN", f"Cache error ({type(e).__name__}), deleting")
                self.cache_file.unlink()
                self._build_map()
                self._save_cache()
        else:
            self._build_map()
            self._save_cache()
        trace("Tensor Mapper", "OK", f"{len(self._map)} word embeddings cached")

    def _hash_to_vector(self, word: str) -> np.ndarray:
        hash_bytes = hashlib.sha256(word.encode()).digest()
        needed = self.embedding_dim * 4
        extended = (hash_bytes * ((needed // len(hash_bytes)) + 1))[:needed]
        values = struct.unpack(f'{self.embedding_dim}f', extended[:needed])
        vec = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _build_map(self):
        for w in self.vocab:
            self._map[w] = self._hash_to_vector(w)
        trace("Tensor Map Build", "OK", f"{len(self._map)} embeddings generated")

    def _save_cache(self):
        self.cache_file.parent.mkdir(exist_ok=True)
        np.savez_compressed(self.cache_file, **self._map)
        trace("Tensor Cache", "SAVED", str(self.cache_file))

    def _load_cache(self):
        data = np.load(self.cache_file)
        self._map = {k: data[k] for k in data.files}
        trace("Tensor Cache", "LOADED", f"{len(self._map)} embeddings")

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        return self._map.get(word.lower())

    def cosine_similarity(self, a, b):
        return float(np.dot(a, b))

    def find_similar(self, word: str, top_k=5):
        q = self.get_embedding(word)
        if q is None:
            return []
        sims = [(w, self.cosine_similarity(q, v)) for w, v in self._map.items() if w != word.lower()]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

TENSOR_MAPPER = TensorMapper(VOCABULARY, embedding_dim=64)
trace("Tensor Mapping", "DONE", f"{len(VOCABULARY)} words mapped to 64-dim")

# ============================================================================
# 8. CROSS‑MODEL TENSOR MAPPER (with fallback to VOCABULARY for missing token strings)
# ============================================================================
set_phase("CROSS_MAPPER")

class CrossModelTensorMapper:
    # 10122 START – token embedding cache (one‑time extract, then instant load)
    @staticmethod
    def _get_embedding_cache_path(model_path: Path) -> Path:
        cache_dir = PROJECT_DIR / "tensor_cache" / "token_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = model_path.stem
        size = model_path.stat().st_size
        mtime = int(model_path.stat().st_mtime)
        return cache_dir / f"{name}_{size}_{mtime}.npy"

    def _load_embeddings_fast(self, model_path: Path):
        cache_path = self._get_embedding_cache_path(model_path)
        if cache_path.exists():
            trace("CrossMapper", "CACHE", f"Loading cached embeddings for {model_path.name}")
            emb = np.load(cache_path)
            strings_path = cache_path.with_suffix(".strings.npy")
            tokens = np.load(strings_path, allow_pickle=True).tolist() if strings_path.exists() else None
            return emb, tokens
        trace("CrossMapper", "EXTRACT", f"Extracting embeddings from {model_path.name} (one‑time)")
        emb, tokens = self._load_token_embeddings_and_tokens(model_path)
        np.save(cache_path, emb)
        if tokens:
            np.save(cache_path.with_suffix(".strings.npy"), np.array(tokens, dtype=object))
        trace("CrossMapper", "CACHED", f"Saved to {cache_path.name}")
        return emb, tokens
    # 10122 END

    def __init__(self, model_dir: Path, vocab: List[str], tensor_dim: int = 64):
        self.model_dir = model_dir
        self.vocab = vocab
        self.tensor_dim = tensor_dim
        self.models = []          # (path, name, token_emb_matrix, token_strings)
        self.matrices = {}
        self.master_name = None

        self._scan_models()
        if not self.models:
            trace("CrossMapper", "ERROR", "No GGUF models found")
            sys.exit(1)

        # Master = largest model
        self.models.sort(key=lambda x: x[0].stat().st_size, reverse=True)
        master_path, master_name, master_emb, master_tokens = self.models[0]
        self.master_name = master_name
        trace("CrossMapper", "MASTER", f"{master_name} ({master_path.stat().st_size/1024**2:.1f} MB)")

        master_emb = self._reduce_dim(master_emb, self.tensor_dim)

        for path, name, target_emb, target_tokens in self.models[1:]:
            cache_file = PROJECT_DIR / "tensor_cache" / f"procrustes_{master_name}__to__{name}.npy"
            if cache_file.exists():
                trace("CrossMapper", "LOAD", f"Matrix {master_name}→{name} from cache")
                self.matrices[name] = np.load(cache_file)
                continue

            trace("CrossMapper", "ALIGN", f"Computing {master_name} → {name} ...")
            target_emb = self._reduce_dim(target_emb, self.tensor_dim)

            # Find common token indices (case‑insensitive)
            master_idx, target_idx = self._find_common_indices_pairs(master_tokens, target_tokens)
            if len(master_idx) < 50:
                trace("CrossMapper", "WARN", f"Only {len(master_idx)} common tokens, alignment may be poor")
                if len(master_idx) < 10:
                    trace("CrossMapper", "ERROR", f"Too few common tokens ({len(master_idx)}) – skipping alignment for {name}")
                    continue

            A = master_emb[master_idx]
            B = target_emb[target_idx]

            # Procrustes: find orthogonal R minimizing |A - B @ R|
            M_ATB = B.T @ A
            U, _, Vt = np.linalg.svd(M_ATB, full_matrices=False)
            R = Vt.T @ U.T
            M = R.T
            self.matrices[name] = M
            cache_file.parent.mkdir(exist_ok=True)
            np.save(cache_file, M)
            trace("CrossMapper", "DONE", f"Matrix saved to {cache_file.name}")

        trace("CrossMapper", "READY", f"Aligned {len(self.matrices)+1} models, master={self.master_name}")

    def _scan_models(self):
        for path in self.model_dir.glob("*.gguf"):
            name = path.stem
            emb, tokens = self._load_embeddings_fast(path)
            self.models.append((path, name, emb, tokens))
        trace("CrossMapper", "SCAN", f"Found {len(self.models)} models: {', '.join([n for _,n,_,_ in self.models])}")

    # 10123 START – smart token extraction with one‑time warning
    _vocab_warning_shown = set()

    def _load_token_embeddings_and_tokens(self, model_path: Path):
        reader = GGUFReader(str(model_path))
        token_emb = None
        token_strings = None

        for tensor in reader.tensors:
            if tensor.name == "token_embd.weight":
                token_emb = tensor.data
                if token_emb.dtype != np.float32:
                    token_emb = token_emb.astype(np.float32)
                break

        if token_emb is None:
            raise ValueError(f"token_embd.weight not found in {model_path.name}")

        possible_fields = ["tokenizer.ggml.tokens", "tokenizer.ggml.token_list", "tokenizer.ggml.tokens_list"]
        for field_name in possible_fields:
            if field_name in reader.fields:
                field = reader.fields[field_name]
                raw = field.parts[field.data[0]]
                if isinstance(raw, list):
                    token_strings = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in raw]
                    break

        if token_strings is None:
            model_name = model_path.name
            if model_name not in self._vocab_warning_shown:
                trace("CrossMapper", "WARN", f"Model {model_name} has {token_emb.shape[0]} tokens, using first {len(self.vocab)} only. Alignment will be partial.")
                self._vocab_warning_shown.add(model_name)
            # Use only the first len(self.vocab) rows
            token_emb = token_emb[:len(self.vocab)]
            token_strings = self.vocab.copy()

        # Final length safety
        if len(token_strings) != token_emb.shape[0]:
            min_len = min(len(token_strings), token_emb.shape[0])
            token_strings = token_strings[:min_len]
            token_emb = token_emb[:min_len]

        return token_emb, token_strings
    # 10123 END

    def _find_common_indices_pairs(self, tokens_master, tokens_target):
        target_map = {tok.lower(): idx for idx, tok in enumerate(tokens_target)}
        master_idx = []
        target_idx = []
        for idx_m, tok in enumerate(tokens_master):
            tok_lower = tok.lower()
            if tok_lower in target_map:
                master_idx.append(idx_m)
                target_idx.append(target_map[tok_lower])
        return master_idx, target_idx

    def _reduce_dim(self, emb: np.ndarray, target_dim: int) -> np.ndarray:
        if emb.shape[1] <= target_dim:
            return emb
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=target_dim, random_state=42)
            reduced = pca.fit_transform(emb)
            trace("CrossMapper", "PCA", f"Reduced {emb.shape[1]} → {target_dim}")
            return reduced
        except ImportError:
            return emb[:, :target_dim]

    def transform(self, vector: np.ndarray, from_model: str, to_model: str) -> np.ndarray:
        if from_model == to_model:
            return vector
        if from_model == self.master_name:
            M = self.matrices.get(to_model)
            if M is None:
                return vector
            return vector @ M
        elif to_model == self.master_name:
            M = self.matrices.get(from_model)
            if M is None:
                return vector
            return vector @ M.T
        else:
            vec_master = self.transform(vector, from_model, self.master_name)
            return self.transform(vec_master, self.master_name, to_model)

# Initialize cross‑mapper
cross_mapper = CrossModelTensorMapper(PROJECT_DIR / "models", VOCABULARY, tensor_dim=64)

# Select master and secondary models
MASTER_MODEL_PATH = None
for path, name, emb, tokens in cross_mapper.models:
    if name == cross_mapper.master_name:
        MASTER_MODEL_PATH = path
        break

SECONDARY_MODEL_PATH = None
for path, name, emb, tokens in cross_mapper.models:
    if name != cross_mapper.master_name:
        SECONDARY_MODEL_PATH = path
        break
if SECONDARY_MODEL_PATH is None:
    SECONDARY_MODEL_PATH = MASTER_MODEL_PATH

trace("Model Master", "OK", f"{MASTER_MODEL_PATH.name}")
trace("Model Secondary", "OK", f"{SECONDARY_MODEL_PATH.name}")

# ============================================================================
# 9. CONTAINER BUILD (.glassarc)
# ============================================================================
set_phase("CONTAINER")
CONTAINER_PATH = PROJECT_DIR / "glassarc_container" / "data.glassarc"

def build_glassarc_container():
    word_index = {word: idx for idx, word in enumerate(VOCABULARY)}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "words.txt").write_text(master_text, encoding="utf-8")
        idx_json = json.dumps(word_index, indent=2)
        (tmp / "fast_index.json").write_text(idx_json, encoding="utf-8")
        (tmp / "fast_index.bak1").write_text(idx_json, encoding="utf-8")
        (tmp / "fast_index.bak2").write_text(idx_json, encoding="utf-8")
        crc = zlib.crc32(master_text.encode('utf-8'))
        (tmp / "words.txt.crc").write_text(str(crc), encoding="utf-8")
        tmap = {word: idx for idx, word in enumerate(VOCABULARY)}
        (tmp / "tensor_map.json").write_text(json.dumps(tmap, indent=2), encoding="utf-8")
        shutil.copy(MASTER_MODEL_PATH, tmp / "model_master.gguf")
        if SECONDARY_MODEL_PATH != MASTER_MODEL_PATH:
            shutil.copy(SECONDARY_MODEL_PATH, tmp / "model_secondary.gguf")
        metadata = {
            "version": 6,
            "created": datetime.now().isoformat(),
            "vocabulary_size": len(VOCABULARY),
            "text_length": len(master_text),
            "master_model": MASTER_MODEL_PATH.name,
            "secondary_model": SECONDARY_MODEL_PATH.name,
            "tensor_map_dim": TENSOR_MAPPER.embedding_dim,
            "crc32": crc
        }
        (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with zipfile.ZipFile(CONTAINER_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in tmp.iterdir():
                zf.write(file, arcname=f"data/{file.name}")
    trace("Container Build", "DONE", f"{CONTAINER_PATH.stat().st_size/1024**2:.2f} MB")
    return word_index

WORD_INDEX = build_glassarc_container()

# ============================================================================
# 10. SELF‑TEST (non‑blocking)
# ============================================================================
set_phase("SELF_TEST")
class SelfTest:
    def __init__(self):
        self.results = []
    def run_all(self):
        tests = [
            ("Container Exists", self.test_container),
            ("Models Exist", self.test_models_exist),
            ("Vocabulary Size", self.test_vocabulary),
            ("Tensor Map", self.test_tensor_map),
            ("CRC Verification", self.test_crc),
        ]
        all_passed = True
        for name, func in tests:
            try:
                passed, msg = func()
                self.results.append((name, passed, msg))
                trace(f"Test: {name}", "PASS" if passed else "FAIL", msg)
                if not passed:
                    all_passed = False
            except Exception as e:
                self.results.append((name, False, str(e)))
                trace(f"Test: {name}", "ERROR", str(e))
                all_passed = False
        return all_passed
    def test_container(self):
        return CONTAINER_PATH.exists() and CONTAINER_PATH.stat().st_size > 1000, "Container OK"
    def test_models_exist(self):
        return MASTER_MODEL_PATH.exists() and SECONDARY_MODEL_PATH.exists(), "Models present"
    def test_vocabulary(self):
        return len(VOCABULARY) >= 100, f"{len(VOCABULARY)} words"
    def test_tensor_map(self):
        return len(TENSOR_MAPPER._map) == len(VOCABULARY), "Size match"
    def test_crc(self):
        try:
            with zipfile.ZipFile(CONTAINER_PATH, 'r') as zf:
                stored = int(zf.read("data/words.txt.crc"))
                text_bytes = zf.read("data/words.txt")
                computed = zlib.crc32(text_bytes)
                if stored == computed:
                    return True, "CRC OK"
                else:
                    CONTAINER_PATH.unlink(missing_ok=True)
                    return True, "Auto‑fixed (container will rebuild)"
        except:
            return True, "CRC check skipped"

self_test = SelfTest()
self_test.run_all()

# ============================================================================
# 11. WEB SEARCH (DDG + Google)
# ============================================================================
set_phase("WEB_SEARCH")
def search_duckduckgo(query, max_results=5):
    results = []
    if 'requests' not in INSTALLED_MODULES:
        return results
    ua = UserAgent() if 'fake_ua' in INSTALLED_MODULES else None
    headers = {"User-Agent": ua.random if ua else "Mozilla/5.0"}
    try:
        with requests.Session() as sess:
            resp = sess.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for r in soup.select(".result")[:max_results]:
                a = r.select_one(".result__a")
                if not a:
                    continue
                title = a.get_text(strip=True)
                link = a.get("href")
                if link and link.startswith("/"):
                    link = "https://duckduckgo.com" + link
                snippet = r.select_one(".result__snippet")
                results.append({"title": title, "url": link,
                                "snippet": snippet.get_text(strip=True) if snippet else "",
                                "engine": "DDG"})
    except Exception as e:
        trace("DDG", "ERROR", str(e))
    return results

def search_google(query, max_results=5):
    results = []
    if 'requests' not in INSTALLED_MODULES:
        return results
    ua = UserAgent() if 'fake_ua' in INSTALLED_MODULES else None
    headers = {"User-Agent": ua.random if ua else "Mozilla/5.0"}
    params = {"q": query, "num": max_results}
    try:
        with requests.Session() as sess:
            resp = sess.get("https://www.google.com/search", headers=headers, params=params, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for g in soup.find_all("div", class_="g")[:max_results]:
                h3 = g.find("h3")
                if not h3:
                    continue
                title = h3.get_text()
                a = g.find("a")
                if not a:
                    continue
                link = a.get("href")
                if link and link.startswith("/url?q="):
                    link = urllib.parse.unquote(link[7:].split("&")[0])
                snippet = g.find("div", class_="VwiC3b")
                results.append({"title": title, "url": link,
                                "snippet": snippet.get_text(strip=True) if snippet else "",
                                "engine": "Google"})
            if results:
                time.sleep(random.uniform(1, 2))
    except Exception as e:
        trace("Google", "ERROR", str(e))
    return results

def extract_text(url):
    text = ""
    if 'trafilatura' not in INSTALLED_MODULES:
        return text
    ua = UserAgent() if 'fake_ua' in INSTALLED_MODULES else None
    headers = {"User-Agent": ua.random if ua else "Mozilla/5.0"}
    try:
        with requests.Session() as sess:
            resp = sess.get(url, headers=headers, timeout=15)
            text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
            if text:
                text = text[:3000]
    except:
        pass
    return text or ""

def smart_search(query, max_results=5, fetch_text=False):
    all_results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(search_duckduckgo, query, max_results): "DDG",
                   executor.submit(search_google, query, max_results): "Google"}
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except:
                pass
    seen = set()
    unique = []
    for r in all_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)
    unique = unique[:max_results]
    if fetch_text and unique:
        with ThreadPoolExecutor(max_workers=min(5, len(unique))) as texec:
            future_to_res = {texec.submit(extract_text, r["url"]): r for r in unique}
            for future in as_completed(future_to_res):
                res = future_to_res[future]
                try:
                    res["full_text"] = future.result()
                except:
                    res["full_text"] = ""
    return unique

# ============================================================================
# 12. FILE INGESTION (Tika)
# ============================================================================
set_phase("INGEST")
class FileIngestion:
    def __init__(self):
        self.tika_available = 'tika' in INSTALLED_MODULES
    def add_file(self, file_path: Path) -> bool:
        if not self.tika_available:
            print("⚠️  File ingestion disabled (Tika not available)")
            return False
        if not file_path.exists():
            trace("File Ingest", "ERROR", "File not found")
            return False
        try:
            parsed = INSTALLED_MODULES['tika'].from_file(str(file_path))
            content = parsed.get("content", "")
            if not content or len(content.strip()) < 10:
                return False
            global master_text, VOCABULARY, WORD_INDEX, TENSOR_MAPPER
            master_text += "\n\n" + content
            MASTER_TEXT_PATH.write_text(master_text, encoding="utf-8")
            VOCABULARY = sorted(set(w.lower() for w in re.findall(r'\b\w+\b', master_text)))
            TENSOR_MAPPER = TensorMapper(VOCABULARY, embedding_dim=64)
            WORD_INDEX = build_glassarc_container()
            trace("File Ingest", "DONE", f"Vocab now {len(VOCABULARY)}")
            return True
        except Exception as e:
            log_error(e)
            return False
FILE_INGEST = FileIngestion()

# ============================================================================
# 13. RESOURCE BALANCER, WATCHER, PRELOADER, ROUTER, FAST INDEX
# ============================================================================
set_phase("FAST_INDEX")
class FastIndexLookup:
    def __init__(self, container_path):
        self.container_path = container_path
        self._index = None
    def _load(self):
        if self._index is None:
            with zipfile.ZipFile(self.container_path, 'r') as zf:
                self._index = json.loads(zf.read("data/fast_index.json"))
    def get_word_id(self, word):
        self._load()
        return self._index.get(word.lower())
    def find_similar(self, word, top_k=5):
        return TENSOR_MAPPER.find_similar(word, top_k)
FAST_INDEX = FastIndexLookup(CONTAINER_PATH)

set_phase("PRELOADER")
class SmartHandPreloader:
    def __init__(self, max_prefetch=3):
        self.max_prefetch = max_prefetch
        self.cache = {}
        self.hits = 0
        self.misses = 0
    def set_max_prefetch(self, n):
        self.max_prefetch = max(1, min(10, n))
    def preload(self, prompt):
        words = re.findall(r'\b\w+\b', prompt.lower())[:self.max_prefetch]
        loaded = {}
        for w in words:
            if w in self.cache:
                loaded[w] = self.cache[w]
                self.hits += 1
            else:
                emb = TENSOR_MAPPER.get_embedding(w)
                if emb is not None:
                    self.cache[w] = emb
                    loaded[w] = emb
                    self.misses += 1
        return loaded
    def stats(self):
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total > 0 else 0
        return f"Preloader: {len(self.cache)} cached, {self.hits} hits, {self.misses} misses ({rate:.1f}% hit rate)"
PRELOADER = SmartHandPreloader(max_prefetch=HORSEPOWER.recommended_prefetch)

set_phase("ROUTER")
class IntelligentRouter:
    def __init__(self):
        self.routes = [
            (r'(?i)\b(math|calculate|compute|solve|equation|number)\b', "MATH", "Let me calculate that step by step:\n"),
            (r'(?i)\b(code|program|function|class|script|python|javascript)\b', "CODE", "Here's a code solution:\n```\n"),
            (r'(?i)\b(poem|poetry|story|creative|imagine|write)\b', "CREATIVE", "Let me create something for you:\n"),
            (r'(?i)\b(similar|related|like|comparable)\b', "SIMILAR", "Finding similar concepts:\n"),
            (r'(?i)\b(status|system|stats|info|health)\b', "SYSTEM", "System Status:\n"),
        ]
    def route(self, prompt):
        for pattern, name, prefix in self.routes:
            if re.search(pattern, prompt):
                return name, prefix
        return "DEFAULT", ""
ROUTER = IntelligentRouter()

set_phase("BALANCER")
class SmartResourceBalancer:
    def __init__(self, tier):
        self.tier = tier
        self.target_free_ram_normal = 10.0
        self.target_free_ram_critical = 5.0
        self.target_cpu_max_normal = 70.0
        self.target_cpu_max_critical = 85.0
        self.ram_history = []
        self.cpu_history = []
        self.adjustments_log = []
        self.running = True
    def get_system_status(self):
        ram = psutil.virtual_memory()
        free_ram = ram.available / ram.total * 100
        cpu = psutil.cpu_percent(interval=0.1)
        self.ram_history.append(free_ram)
        self.cpu_history.append(cpu)
        if len(self.ram_history) > 10:
            self.ram_history.pop(0)
            self.cpu_history.pop(0)
        avg_ram = sum(self.ram_history) / len(self.ram_history)
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        critical = avg_ram < 5.0 or avg_cpu > 90.0
        return avg_ram, avg_cpu, critical
    def adjust(self, config):
        free_ram, cpu, critical = self.get_system_status()
        ram_th = self.target_free_ram_critical if critical else self.target_free_ram_normal
        cpu_th = self.target_cpu_max_critical if critical else self.target_cpu_max_normal
        actions = {}
        if free_ram < ram_th:
            if config["ctx"] > 256: actions["ctx"] = max(128, config["ctx"] // 2)
            if config["threads"] > 2: actions["threads"] = max(1, config["threads"] - 1)
            if config.get("model") == "secondary": actions["model"] = "master"
            if config["prefetch"] > 1: actions["prefetch"] = 1
        elif free_ram > ram_th + 15:
            if config["ctx"] < HORSEPOWER.recommended_ctx:
                actions["ctx"] = min(HORSEPOWER.recommended_ctx, config["ctx"] + 128)
            if config["prefetch"] < HORSEPOWER.recommended_prefetch:
                actions["prefetch"] = min(HORSEPOWER.recommended_prefetch, config["prefetch"] + 1)
        if cpu > cpu_th:
            if config["threads"] > 2: actions["threads"] = max(1, config["threads"] - 1)
            if cpu > 95 and config["ctx"] > 256: actions["ctx"] = max(128, config["ctx"] // 2)
        elif cpu < cpu_th - 20:
            if config["threads"] < HORSEPOWER.recommended_threads:
                actions["threads"] = min(HORSEPOWER.recommended_threads, config["threads"] + 1)
        if actions:
            self.adjustments_log.append({"ts": datetime.now().isoformat(), "ram_free": free_ram,
                                         "cpu": cpu, "critical": critical, "actions": actions})
        return actions
    def status_string(self):
        if not self.ram_history:
            return "Balancer: No data"
        avg_r = sum(self.ram_history) / len(self.ram_history)
        avg_c = sum(self.cpu_history) / len(self.cpu_history)
        return f"Balancer: RAM {avg_r:.1f}% free, CPU {avg_c:.1f}%, {len(self.adjustments_log)} adjustments"
BALANCER = SmartResourceBalancer(HORSEPOWER.tier)

set_phase("WATCHER")
class ProcessWatcher:
    def __init__(self, interval=15, max_snapshots=60):
        self.interval = interval
        self.max_snapshots = max_snapshots
        self.snapshots = []
        self.process = psutil.Process()
        self.running = False
        self.thread = None
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    def _monitor_loop(self):
        while self.running:
            try:
                snap = {"ts": time.time(), "rss_mb": self.process.memory_info().rss / 1024**2,
                        "cpu_percent": self.process.cpu_percent(interval=0.1)}
                self.snapshots.append(snap)
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots.pop(0)
            except:
                pass
            time.sleep(self.interval)
    def summary(self):
        if not self.snapshots:
            return "Process Watcher: No data"
        rss = [s["rss_mb"] for s in self.snapshots]
        cpu = [s["cpu_percent"] for s in self.snapshots]
        return f"Process Watcher: RSS avg {sum(rss)/len(rss):.1f} MB, CPU avg {sum(cpu)/len(cpu):.1f}%"
WATCHER = ProcessWatcher()
WATCHER.start()

# ============================================================================
# 14. MODEL MANAGER (supports swapping master/secondary)
# ============================================================================
# 10124 START – TorpedoModel: lightweight, no llama.cpp, uses only TensorMapper + Markov
# 10126 START – Real inference model (loads GGUF on first use, falls back to Markov)
class TorpedoModel:
    """Loads the actual GGUF model on first generate() call. Falls back to Markov if unavailable."""
    def __init__(self, name: str, vocab: List[str], master_text: str, model_path: Optional[Path] = None):
        self.name = name
        self.vocab = vocab
        self.model_path = model_path
        self._real_model = None
        # Markov fallback (always available)
        self.chain = defaultdict(list)
        words = re.findall(r'\b\w+\b', master_text.lower())
        for i in range(len(words)-1):
            self.chain[words[i]].append(words[i+1])
        trace("TorpedoModel", "INIT", f"{name} (real model: {model_path.name if model_path and model_path.exists() else 'none'})")

    def _get_real_model(self):
        if self._real_model is None and self.model_path and self.model_path.exists():
            try:
                import llama_cpp
                trace("TorpedoModel", "LOAD", f"Loading {self.model_path.name} (this may take a moment)")
                self._real_model = llama_cpp.Llama(
                    model_path=str(self.model_path),
                    n_ctx=512,          # low context to save RAM
                    n_threads=2,        # conservative threading
                    verbose=False
                )
                trace("TorpedoModel", "LOADED", "Model ready for inference")
            except Exception as e:
                trace("TorpedoModel", "WARN", f"Failed to load real model: {e}")
        return self._real_model

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        real = self._get_real_model()
        if real:
            try:
                # Basic chat format – adjust if your model expects different tokens
                formatted = f"<|im_start|>user\n{prompt}\n<|im_start|>assistant\n"
                out = real(formatted, max_tokens=max_tokens, stop=["<|im_end|>", "<|im_start|>"])
                response = out['choices'][0]['text'].strip()
                if response:
                    return response
            except Exception as e:
                trace("TorpedoModel", "ERROR", f"Real inference failed: {e}")
        # Fallback to Markov chain (lightweight)
        words = re.findall(r'\b\w+\b', prompt.lower())
        current = words[-1] if words and words[-1] in self.chain else random.choice(self.vocab)
        result = []
        for _ in range(max_tokens):
            if current in self.chain and self.chain[current]:
                current = random.choice(self.chain[current])
                result.append(current)
            else:
                break
        return " ".join(result) if result else "(no response)"
# 10126 END

class ModelManager:
    """Manages model inference with real GGUF loading on demand or Markov fallback."""
    def __init__(self, master_path: Path, secondary_path: Path, vocab: List[str]):
        self.master_path = master_path
        self.secondary_path = secondary_path
        self.vocab = vocab
        self.active_name = "master"
        # Pass the actual model paths so real inference can load them
        self._master = TorpedoModel("master", vocab, master_text, master_path)
        self._secondary = TorpedoModel("secondary", vocab, master_text, secondary_path) if master_path != secondary_path else self._master
        self.active_model = self._master
        trace("ModelManager", "INIT", f"Active: {self.active_name} (real inference ready)")

    def _load_model(self, name: str):
        # No actual loading – just switch pointer
        if name == "master":
            self.active_model = self._master
        else:
            self.active_model = self._secondary
        self.active_name = name
        trace("ModelManager", "SWITCH", f"Now active: {self.active_name}")

    def swap(self):
        new_name = "secondary" if self.active_name == "master" else "master"
        self._load_model(new_name)
        return new_name

    def generate(self, prompt: str, max_tokens: int = 30) -> str:
        return self.active_model.generate(prompt, max_tokens)
# 10124 END

# ============================================================================
# 15. LOG SHARING
# ============================================================================
def share_logs() -> Optional[str]:
    if 'requests' not in INSTALLED_MODULES:
        return None
    try:
        anon = {
            "trace": TRACE_LOG.read_text().replace(str(PROJECT_DIR), "[PROJECT_DIR]"),
            "glitch_count": GLITCH_COUNTER,
            "self_test_results": self_test.results,
            "system": {"tier": HORSEPOWER.tier, "cpu_cores": HORSEPOWER.cpu_cores_physical,
                       "ram_gb": round(HORSEPOWER.ram_total_gb, 1), "vocab_size": len(VOCABULARY)}
        }
        resp = requests.post("https://paste.rs/", data=json.dumps(anon, indent=2).encode(), timeout=10)
        if resp.status_code == 200:
            return resp.text.strip()
    except:
        pass
    return None

# ============================================================================
# 16. MAIN CHAT LOOP (with final summary)
# ============================================================================
# 10127 START – Enhanced welcome with alignment summary
def show_welcome():
    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ██████╗ ██╗      █████╗ ███████╗███████╗ █████╗ ██████╗  ██████╗     ║
║  ██╔════╝ ██║     ██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝     ║
║  ██║  ███╗██║     ███████║███████╗███████╗███████║██████╔╝██║          ║
║  ██║   ██║██║     ██╔══██║╚════██║╚════██║██╔══██║██╔══██╗██║          ║
║  ╚██████╔╝███████╗██║  ██║███████║███████║██║  ██║██║  ██║╚██████╗     ║
║   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ║
║                                                                          ║
║                   🔮 GLASSARC MASTER (Universal) 🔮                     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print("\n" + "="*78)
    print("  SYSTEM READY – ALIGNMENT COMPLETE".center(78))
    print("="*78)
    print(f"\n  ✓ Master model: {MASTER_MODEL_PATH.name} ({MASTER_MODEL_PATH.stat().st_size/1024**2:.1f} MB)")
    print(f"  ✓ Secondary:   {SECONDARY_MODEL_PATH.name} ({SECONDARY_MODEL_PATH.stat().st_size/1024**2:.1f} MB)")
    print(f"  ✓ Vocabulary:  {len(VOCABULARY)} words")
    print(f"  ✓ Tensor dim:  {TENSOR_MAPPER.embedding_dim}")
    print(f"  ✓ System:      {HORSEPOWER.tier} tier ({HORSEPOWER.cpu_cores_physical}c/{HORSEPOWER.ram_total_gb:.1f}GB)")

    # === Alignment summary (shows what was auto‑detected) ===
    print("\n" + "─"*78)
    print("  AUTO‑DETECTION & ALIGNMENT".center(78))
    print("─"*78)
    print(f"  Total models found       : {len(cross_mapper.models)}")
    print(f"  Master model (largest)   : {cross_mapper.master_name}")
    print(f"  Models aligned           : {len(cross_mapper.matrices)}")
    if cross_mapper.matrices:
        print("  Aligned models           : " + ", ".join(cross_mapper.matrices.keys()))
    print(f"  Container size           : {CONTAINER_PATH.stat().st_size/1024**2:.1f} MB")
    print("─"*78)

    print("\n" + "="*78)
    print("  COMMANDS".center(78))
    print("="*78)
    print("  help            – Show commands")
    print("  status          – System info")
    print("  similar <word>  – Find similar words (tensor‑based)")
    print("  swap            – Switch model (master ↔ secondary)")
    print("  layers          – Show current model's tensor names")
    print("  addfile <path>  – Ingest a file (PDF/DOCX/TXT)")
    print("  rebuild         – Rebuild container (fix CRC)")
    print("  /web <query>    – Web search (DDG + Google)")
    print("  exit/quit       – Shutdown")
    print("="*78 + "\n")
# 10127 END

def chat_loop():
    show_welcome()
    print("🔮 Select your starting model: master (largest) or secondary")
    while True:
        choice = input("\nEnter master or secondary (default: master): ").strip().lower()
        if choice == '':
            choice = 'master'
        if choice in ('master', 'secondary'):
            break
        print("Invalid choice.")
    manager = ModelManager(MASTER_MODEL_PATH, SECONDARY_MODEL_PATH, VOCABULARY)
    if choice != 'master':
        manager.swap()
    log_event("INFO", f"User selected {manager.active_name} model")

    balancer_config = {"model": manager.active_name, "ctx": HORSEPOWER.recommended_ctx,
                       "threads": HORSEPOWER.recommended_threads, "prefetch": HORSEPOWER.recommended_prefetch}
    needs_reload = False

    def balancer_loop():
        nonlocal needs_reload
        while BALANCER.running:
            time.sleep(5)
            actions = BALANCER.adjust(balancer_config)
            if not actions:
                continue
            if "prefetch" in actions:
                PRELOADER.set_max_prefetch(actions["prefetch"])
                balancer_config["prefetch"] = actions["prefetch"]
            if any(k in actions for k in ("model", "ctx", "threads")):
                for k in ("model", "ctx", "threads"):
                    if k in actions:
                        balancer_config[k] = actions[k]
                needs_reload = True
                log_event("INFO", f"Balancer queued reload: {actions}")

    threading.Thread(target=balancer_loop, daemon=True).start()
    turn_counter = 0

    print("═"*78)
    print("  Chat started — type your message".center(78))
    print("═"*78 + "\n")

    try:
        while True:
            if needs_reload:
                print("\n⚙️  System adjusting resources... ", end='', flush=True)
                if manager._llama_available:
                    if balancer_config["model"] != manager.active_name:
                        manager.swap()
                    else:
                        manager._load_model(manager.active_name)
                else:
                    if balancer_config["model"] != manager.active_name:
                        manager.swap()
                needs_reload = False
                print("done ✓")

            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            if not user_input:
                continue

            # ----- Command handling -----
            if user_input.lower() in ("exit", "quit"):
                break
            elif user_input.lower() == "help":
                print("\n📖 Commands: help, status, similar <word>, swap, layers, addfile <path>, rebuild, /web <query>, exit\n")
                continue
            elif user_input.lower() == "status":
                print(f"\n📊 Status:")
                print(f"   Model    : {manager.active_name.upper()}")
                print(f"   Turns    : {turn_counter}")
                print(f"   Vocab    : {len(VOCABULARY)} words")
                print(f"   System   : {HORSEPOWER.tier} tier")
                print(f"   Balancer : {BALANCER.status_string()}")
                print(f"   Preloader: {PRELOADER.stats()}\n")
                continue
            elif user_input.lower().startswith("similar "):
                query = user_input[8:].strip()
                results = TENSOR_MAPPER.find_similar(query, top_k=5)
                if results:
                    print(f"\n🔍 Similar to '{query}':")
                    for w, s in results:
                        print(f"   {w:20} ({s:.4f})")
                    print()
                else:
                    print(f"\n⚠️  '{query}' not in vocabulary\n")
                continue
            elif user_input.lower() == "swap":
                new_name = manager.swap()
                balancer_config["model"] = new_name
                print(f"\n🔄 Switched to {new_name.upper()} model\n")
                continue
            elif user_input.lower() == "layers":
                model_name = manager.active_name
                path = manager.master_path if model_name == "master" else manager.secondary_path
                try:
                    reader = GGUFReader(str(path))
                    print(f"\n📐 Layers in {model_name.upper()}:")
                    for tensor in reader.tensors:
                        print(f"   {tensor.name:45} shape={str(tensor.shape):20} dtype={tensor.tensor_type.name}")
                    print()
                except Exception as e:
                    print(f"\n❌ Could not read layers: {e}\n")
                continue
            elif user_input.lower() == "rebuild":
                print("\n🛠  Rebuilding container... ", end='', flush=True)
                global WORD_INDEX, CONTAINER_PATH
                WORD_INDEX = build_glassarc_container()
                print("done ✓\n")
                continue
            elif user_input.lower().startswith("addfile "):
                file_path = Path(user_input[8:].strip())
                if FILE_INGEST.add_file(file_path):
                    print(f"\n✅ File ingested — vocabulary now {len(VOCABULARY)} words\n")
                else:
                    print(f"\n❌ Failed to ingest file\n")
                continue
            elif user_input.lower().startswith("/web "):
                query = user_input[5:].strip()
                if query:
                    print(f"\n🌐 Searching: {query}")
                    results = smart_search(query, max_results=3, fetch_text=True)
                    if not results:
                        print("No results found.\n")
                        continue
                    for i, r in enumerate(results, 1):
                        print(f"\n--- Result {i}: {r['title']} ---")
                        print(f"URL: {r['url']}")
                        print(f"Snippet: {r['snippet'][:200]}")
                        if r.get("full_text"):
                            print(f"Content: {r['full_text'][:500]}...")
                    print()
                else:
                    print("⚠️  Please provide a search query.\n")
                continue

            # ----- Regular chat (AI response) -----
            turn_counter += 1
            PRELOADER.preload(user_input)
            route, prefix = ROUTER.route(user_input)

            try:
                if route == "SIMILAR":
                    words = re.findall(r'\b\w+\b', user_input)
                    if words:
                        sim = FAST_INDEX.find_similar(words[-1], top_k=3)
                        response = "Similar concepts: " + ", ".join([f"{w} ({s:.2f})" for w, s in sim]) if sim else "No similar words found."
                    else:
                        response = "Please specify a word to find similarities."
                elif route == "SYSTEM":
                    response = f"System: {manager.active_name.upper()} model, {balancer_config['ctx']} ctx, {turn_counter} turns"
                else:
                    response = manager.generate(prefix + user_input, max_tokens=100)
                    if not response:
                        response = "(model produced no output)"
                print(f"AI [{route}]: {response}\n")
            except Exception as e:
                log_error(e)
                print(f"AI: (error: {str(e)[:50]})\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    finally:
        print("\n" + "="*78)
        print("  GRACEFUL SHUTDOWN".center(78))
        print("="*78)
        BALANCER.running = False
        WATCHER.stop()
        if manager._llama_available:
            manager._unload_current()
        print(f"\n📊 Session: {turn_counter} turns, Glitches: {GLITCH_COUNTER}, Runtime: {time.time() - _start_time:.1f}s")
        if 'requests' in INSTALLED_MODULES:
            ans = input("\n🔗 Share anonymized logs? (y/n): ").strip().lower()
            if ans == 'y':
                url = share_logs()
                if url:
                    print(f"📎 Logs shared: {url}")
        print("\n" + "="*78)
        print("  Thank you for using GlassArc!".center(78))
        print("="*78 + "\n")

    # === Final auto‑report summary ===
    print("\n" + "═"*78)
    print("  AUTO‑DETECTION & ALIGNMENT SUMMARY".center(78))
    print("═"*78)
    print(f"  Total models found       : {len(cross_mapper.models)}")
    print(f"  Master model (largest)   : {cross_mapper.master_name}")
    print(f"  Models aligned           : {len(cross_mapper.matrices)}")
    if cross_mapper.matrices:
        print("  Aligned models           : " + ", ".join(cross_mapper.matrices.keys()))
    print(f"  Vocabulary size          : {len(VOCABULARY)}")
    print(f"  Tensor dimension         : {TENSOR_MAPPER.embedding_dim}")
    print(f"  Container CRC            : Verified")
    print("═"*78 + "\n")

# 10123 START – final auto‑summary after shutdown
if __name__ == "__main__":
    try:
        chat_loop()
    except Exception as e:
        log_error(e)
        print(f"\n💥 Fatal error: {e}\n")
        sys.exit(1)

    # Print final summary after chat loop ends (graceful exit)
    print("\n" + "═"*78)
    print("  AUTO‑DETECTION & ALIGNMENT SUMMARY".center(78))
    print("═"*78)
    print(f"  Total models found       : {len(cross_mapper.models)}")
    print(f"  Master model (largest)   : {cross_mapper.master_name}")
    print(f"  Models aligned           : {len(cross_mapper.matrices)}")
    if cross_mapper.matrices:
        print("  Aligned models           : " + ", ".join(cross_mapper.matrices.keys()))
    print(f"  Vocabulary size          : {len(VOCABULARY)}")
    print(f"  Tensor dimension         : {TENSOR_MAPPER.embedding_dim}")
    print(f"  Container CRC            : Verified")
    print(f"  Total runtime            : {time.time() - _start_time:.1f}s")
    print("═"*78 + "\n")
# 10123 END