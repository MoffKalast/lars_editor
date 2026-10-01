import argparse
import json
import mimetypes
import os
import re
import struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

LARS_CATEGORIES = [
	{"id": 1, "name": "Static Obstacle", "supercategory": "obstacle", "isthing": 0, "color": [255, 212, 25]},
	{"id": 3, "name": "Water", "supercategory": "water", "isthing": 0, "color": [70, 245, 255]},
	{"id": 5, "name": "Sky", "supercategory": "sky", "isthing": 0, "color": [170, 0, 255]},
	{"id": 11, "name": "Boat/ship", "supercategory": "obstacle", "isthing": 1, "color": [255, 39, 43]},
	{"id": 12, "name": "Row boats", "supercategory": "obstacle", "isthing": 1, "color": [255, 39, 43]},
	{"id": 13, "name": "Paddle board", "supercategory": "obstacle", "isthing": 1, "color": [255, 39, 43]},
	{"id": 14, "name": "Buoy", "supercategory": "obstacle", "isthing": 1, "color": [255, 39, 43]},
	{"id": 15, "name": "Swimmer", "supercategory": "obstacle", "isthing": 1, "color": [255, 39, 43]},
	{"id": 16, "name": "Animal", "supercategory": "obstacle", "isthing": 1, "color": [255, 39, 43]},
	{"id": 17, "name": "Float", "supercategory": "obstacle", "isthing": 1, "color": [255, 39, 43]},
	{"id": 19, "name": "Other", "supercategory": "obstacle", "isthing": 1, "color": [255, 39, 43]},
]

SEMANTIC_VALUES = {"obstacle": 0, "water": 1, "sky": 2}
IGNORE_SEMANTIC = 255

ROOT = None
SAM_URL = None
CLASS_CACHE = {}


def find_splits(root):
	splits = []

	if (root / "images").is_dir():
		splits.append(".")

	for child in sorted(root.iterdir()):
		if child.is_dir() and (child / "images").is_dir():
			splits.append(child.name)

	return splits


def split_dir(split):
	if split not in find_splits(ROOT):
		raise ValueError(f"unknown split: {split}")

	return ROOT / split


def natural_key(name):
	return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_images(sdir):
	return sorted((p.name for p in (sdir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)


def image_path(sdir, name):
	if name not in list_images(sdir):
		raise ValueError(f"unknown image: {name}")

	return sdir / "images" / name


def mask_name(name):
	return Path(name).stem + ".png"


def load_annotations(sdir):
	path = sdir / "panoptic_annotations.json"

	if path.exists():
		with open(path) as f:
			return json.load(f)

	return {"info": {}, "images": [], "annotations": [], "categories": LARS_CATEGORIES, "licenses": []}


def save_annotations(sdir, data):
	path = sdir / "panoptic_annotations.json"
	tmp = path.with_suffix(".json.tmp")

	with open(tmp, "w") as f:
		json.dump(data, f)

	os.replace(tmp, path)


def find_annotation(data, name):
	target = mask_name(name)

	for ann in data["annotations"]:
		if ann["file_name"] == target:
			return ann

	return None


def read_ids(sdir, name):
	path = sdir / "panoptic_masks" / mask_name(name)

	if not path.exists():
		width, height = Image.open(image_path(sdir, name)).size
		return np.zeros((height, width), dtype=np.uint32)

	rgb = np.array(Image.open(path).convert("RGB")).astype(np.uint32)
	return rgb[..., 0] + (rgb[..., 1] << 8) + (rgb[..., 2] << 16)


def load_meta(sdir, name):
	data = load_annotations(sdir)
	ids = read_ids(sdir, name)
	cat_ids = {c["id"] for c in data["categories"]}
	ann = find_annotation(data, name)
	known = {s["id"]: s for s in ann["segments_info"]} if ann else {}
	segments = []

	for sid in np.unique(ids).tolist():
		if sid == 0:
			continue

		if sid in known:
			segments.append({"id": sid, "category_id": known[sid]["category_id"], "iscrowd": known[sid].get("iscrowd", 0)})
		else:
			category = sid & 0xFFFF if (sid & 0xFFFF) in cat_ids else 19
			segments.append({"id": sid, "category_id": category, "iscrowd": 0})

	height, width = ids.shape
	return {"width": width, "height": height, "categories": data["categories"], "segments": segments}, ids


def mask_classes(path):
	mtime = path.stat().st_mtime
	cached = CLASS_CACHE.get(path)

	if cached and cached[0] == mtime:
		return cached[1]

	rgb = np.array(Image.open(path).convert("RGB")).astype(np.uint32)
	classes = np.nonzero(np.bincount((rgb[..., 0] + (rgb[..., 1] << 8)).ravel()))[0].tolist()
	CLASS_CACHE[path] = (mtime, classes)
	return classes


def class_key(category_id, iscrowd):
	return f"{category_id}c" if iscrowd else str(category_id)


def image_classes(sdir):
	data = load_annotations(sdir)
	cat_ids = {c["id"] for c in data["categories"]}
	known = {ann["file_name"]: sorted({class_key(s["category_id"], s.get("iscrowd", 0)) for s in ann["segments_info"]}) for ann in data["annotations"]}
	result = {}

	for name in list_images(sdir):
		path = sdir / "panoptic_masks" / mask_name(name)

		if mask_name(name) in known:
			result[name] = known[mask_name(name)]
		elif path.exists():
			result[name] = sorted({class_key(c if c in cat_ids else 19, 0) for c in mask_classes(path) if c != 0})
		else:
			result[name] = []

	return {"categories": data["categories"], "images": result}


def save_image(sdir, name, ids, segment_map):
	data = load_annotations(sdir)
	categories = {c["id"]: c for c in data["categories"]}
	height, width = ids.shape

	unique, inverse = np.unique(ids, return_inverse=True)
	inverse = inverse.reshape(ids.shape)
	semantic_lut = np.full(len(unique), IGNORE_SEMANTIC, dtype=np.uint8)
	segments_info = []

	for index, sid in enumerate(unique.tolist()):
		if sid == 0:
			continue

		segment = segment_map[sid]
		category = categories[segment["category_id"]]
		semantic_lut[index] = SEMANTIC_VALUES[category["supercategory"]]

		ys, xs = np.nonzero(inverse == index)
		x0, y0 = int(xs.min()), int(ys.min())
		bbox = [x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1]
		segments_info.append({"id": sid, "category_id": segment["category_id"], "area": int(len(xs)), "bbox": bbox, "iscrowd": int(segment["iscrowd"])})

	rgb = np.stack([ids & 255, (ids >> 8) & 255, (ids >> 16) & 255], axis=-1).astype(np.uint8)
	(sdir / "panoptic_masks").mkdir(exist_ok=True)
	(sdir / "semantic_masks").mkdir(exist_ok=True)
	Image.fromarray(rgb, "RGB").save(sdir / "panoptic_masks" / mask_name(name))
	Image.fromarray(semantic_lut[inverse], "L").save(sdir / "semantic_masks" / mask_name(name))

	ann = find_annotation(data, name)

	if ann is None:
		image_id = max((img["id"] for img in data["images"]), default=0) + 1
		data["images"].append({"id": image_id, "width": width, "height": height, "file_name": name})
		data["annotations"].append({"image_id": image_id, "file_name": mask_name(name), "segments_info": segments_info})
	else:
		ann["segments_info"] = segments_info

	save_annotations(sdir, data)

	list_path = sdir / "image_list.txt"
	stems = list_path.read_text().split() if list_path.exists() else []

	if Path(name).stem not in stems:
		with open(list_path, "a") as f:
			f.write(Path(name).stem + "\n")


class Handler(BaseHTTPRequestHandler):
	def log_message(self, format, *args):
		pass

	def send(self, status, body, content_type="application/json", headers=None):
		if isinstance(body, (dict, list)):
			body = json.dumps(body).encode()

		self.send_response(status)
		self.send_header("Content-Type", content_type)
		self.send_header("Content-Length", str(len(body)))
		self.send_header("Cache-Control", "no-store")

		for key, value in (headers or {}).items():
			self.send_header(key, value)

		self.end_headers()
		self.wfile.write(body)

	def params(self):
		query = parse_qs(urlparse(self.path).query)
		return {key: values[0] for key, values in query.items()}

	def do_GET(self):
		route = urlparse(self.path).path
		p = self.params()

		try:
			if route == "/":
				self.send(200, (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
			elif route == "/api/splits":
				self.send(200, {"root": str(ROOT), "splits": find_splits(ROOT)})
			elif route == "/api/images":
				sdir = split_dir(p["split"])
				masks = sdir / "panoptic_masks"
				done = {f.name for f in masks.iterdir()} if masks.is_dir() else set()
				self.send(200, [{"name": n, "labeled": mask_name(n) in done} for n in list_images(sdir)])
			elif route == "/api/classes":
				self.send(200, image_classes(split_dir(p["split"])))
			elif route == "/api/image":
				path = image_path(split_dir(p["split"]), p["name"])
				self.send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
			elif route == "/api/meta":
				meta, _ = load_meta(split_dir(p["split"]), p["name"])
				self.send(200, meta)
			elif route == "/api/ids":
				sdir = split_dir(p["split"])
				image_path(sdir, p["name"])
				ids = read_ids(sdir, p["name"])
				self.send(200, ids.astype("<u4").tobytes(), "application/octet-stream")
			else:
				self.send(404, {"error": "not found"})
		except (ValueError, KeyError) as e:
			self.send(400, {"error": str(e)})

	def sam(self, p):
		path = image_path(split_dir(p["split"]), p["name"])
		query = urlparse(self.path).query

		try:
			with urlopen(Request(f"{SAM_URL}/segment?{query}", data=path.read_bytes()), timeout=120) as response:
				self.send(200, response.read(), "application/octet-stream")
		except HTTPError as e:
			self.send(e.code, e.read())
		except URLError as e:
			self.send(502, {"error": f"cannot reach SAM server at {SAM_URL}: {e.reason}"})

	def do_POST(self):
		route = urlparse(self.path).path
		p = self.params()

		try:
			if route == "/api/sam":
				self.sam(p)
				return

			if route != "/api/save":
				self.send(404, {"error": "not found"})
				return

			sdir = split_dir(p["split"])
			image_path(sdir, p["name"])
			body = self.rfile.read(int(self.headers["Content-Length"]))
			meta_length = struct.unpack("<I", body[:4])[0]
			meta = json.loads(body[4:4 + meta_length])
			ids = np.frombuffer(body[4 + meta_length:], dtype="<u4").reshape(meta["height"], meta["width"])
			segment_map = {s["id"]: s for s in meta["segments"]}

			missing = set(np.unique(ids).tolist()) - set(segment_map) - {0}

			if missing:
				raise ValueError(f"segments without category: {sorted(missing)}")

			save_image(sdir, p["name"], ids, segment_map)
			self.send(200, {"ok": True})
		except (ValueError, KeyError) as e:
			self.send(400, {"error": str(e)})


def main():
	global ROOT, SAM_URL
	parser = argparse.ArgumentParser()
	parser.add_argument("root", nargs="?", default="LaRS_v1.1.0")
	parser.add_argument("--port", type=int, default=8000)
	parser.add_argument("--sam", default="http://127.0.0.1:8001")
	args = parser.parse_args()

	ROOT = Path(args.root).resolve()
	SAM_URL = args.sam

	if not find_splits(ROOT):
		raise SystemExit(f"no folder with an images/ subfolder found in {ROOT}")

	print(f"http://localhost:{args.port}")
	ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
	main()
