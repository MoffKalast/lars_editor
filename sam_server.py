import argparse
import io
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import torch
import torch.nn.functional as F
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

MASK_SIZE = 288
MAX_LABELS = 255

PROCESSOR = None
DTYPE = None
CACHE = {"key": None, "size": None, "state": None}


def segment(data, box, threshold):
	if data != CACHE["key"]:
		CACHE.update(key=None, size=None, state=None)
		image = Image.open(io.BytesIO(data)).convert("RGB")
		CACHE.update(key=data, size=image.size, state=PROCESSOR.set_image(image))
	else:
		PROCESSOR.reset_all_prompts(CACHE["state"])

	width, height = CACHE["size"]
	state = CACHE["state"]
	state["original_width"] = state["original_height"] = MASK_SIZE
	PROCESSOR.confidence_threshold = threshold

	x0, y0, x1, y1 = box
	PROCESSOR.add_geometric_prompt([(x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height], True, state)

	order = state["scores"].argsort()[-MAX_LABELS:].tolist()
	labels = torch.zeros((height, width), dtype=torch.uint8, device=state["scores"].device)

	for label, index in enumerate(order, 1):
		mask = F.interpolate(state["masks_logits"][index:index + 1].float(), size=(height, width), mode="bilinear", align_corners=False)[0, 0]
		labels[mask > 0.5] = label

	return labels.cpu().numpy().tobytes(), len(order)


class Handler(BaseHTTPRequestHandler):
	def log_message(self, format, *args):
		pass

	def send(self, status, body, content_type="application/json"):
		if isinstance(body, dict):
			body = json.dumps(body).encode()

		self.send_response(status)
		self.send_header("Content-Type", content_type)
		self.send_header("Content-Length", str(len(body)))
		self.end_headers()
		self.wfile.write(body)

	def do_POST(self):
		url = urlparse(self.path)

		if url.path != "/segment":
			self.send(404, {"error": "not found"})
			return

		try:
			p = {key: values[0] for key, values in parse_qs(url.query).items()}
			data = self.rfile.read(int(self.headers["Content-Length"]))
			box = [float(v) for v in p["box"].split(",")]
			start = time.time()
			torch.cuda.reset_peak_memory_stats()

			with torch.autocast("cuda", dtype=DTYPE):
				labels, count = segment(data, box, float(p["threshold"]))

			print(f"{p.get('name', '?')}: {count} instances, {time.time() - start:.2f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
			self.send(200, labels, "application/octet-stream")
		except Exception as e:
			traceback.print_exc()
			torch.cuda.empty_cache()
			self.send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
	global PROCESSOR, DTYPE
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", default="sam3.pt")
	parser.add_argument("--port", type=int, default=8001)
	args = parser.parse_args()

	DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
	PROCESSOR = Sam3Processor(build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False))
	print(f"{torch.cuda.get_device_name()}, {DTYPE}, model uses {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
	print(f"http://localhost:{args.port}")
	HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
	main()
