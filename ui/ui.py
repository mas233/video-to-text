#!/usr/bin/env python3
"""
Flask Web UI for video summary pipeline.

功能：
1) 模型检测（设备/模型/dtype 与自动选择逻辑一致性）
2) 环境检查（ffmpeg/ffprobe 与关键 Python 库）
3) 视频上传（<=2GB）+ 执行 + 实时日志 + 产物下载
"""

import contextlib
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

# 允许从项目根目录导入 core 模块
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from core.audio_extractor import AudioExtractor
from core import whisper_parser_v2 as whisper


app = Flask(
	__name__,
	template_folder=str(Path(__file__).resolve().parent / "templates"),
	static_folder=str(Path(__file__).resolve().parent / "static"),
)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
WEBUI_OUTPUT_DIR = ROOT_DIR / "output" / "webui"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WEBUI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}

_checks_lock = threading.Lock()
_model_checks: Dict[str, Dict[str, Any]] = {}


def _now_str() -> str:
	return datetime.now().strftime("%H:%M:%S")


def _append_log(job_id: str, message: str, level: str = "info") -> None:
	line = {
		"ts": _now_str(),
		"level": level,
		"message": message.rstrip("\n"),
	}
	with _jobs_lock:
		job = _jobs.get(job_id)
		if not job:
			return
		job["logs"].append(line)


def _append_check_log(check_id: str, message: str, level: str = "info") -> None:
	line = {
		"ts": _now_str(),
		"level": level,
		"message": message.rstrip("\n"),
	}
	with _checks_lock:
		check = _model_checks.get(check_id)
		if not check:
			return
		check["logs"].append(line)


class _TeeToJob(io.TextIOBase):
	def __init__(self, job_id: str, original_stream: io.TextIOBase, level: str = "info"):
		self.job_id = job_id
		self.original_stream = original_stream
		self.level = level
		self._buffer = ""

	def write(self, s: str) -> int:
		if not isinstance(s, str):
			s = str(s)
		if self.original_stream:
			self.original_stream.write(s)
			self.original_stream.flush()

		self._buffer += s
		while "\n" in self._buffer:
			line, self._buffer = self._buffer.split("\n", 1)
			if line.strip():
				_append_log(self.job_id, line, self.level)
		return len(s)

	def flush(self) -> None:
		if self.original_stream:
			self.original_stream.flush()
		if self._buffer.strip():
			_append_log(self.job_id, self._buffer, self.level)
		self._buffer = ""


class _TeeToCheck(io.TextIOBase):
	def __init__(self, check_id: str, original_stream: io.TextIOBase, level: str = "info"):
		self.check_id = check_id
		self.original_stream = original_stream
		self.level = level
		self._buffer = ""

	def write(self, s: str) -> int:
		if not isinstance(s, str):
			s = str(s)
		if self.original_stream:
			self.original_stream.write(s)
			self.original_stream.flush()

		self._buffer += s
		while "\n" in self._buffer:
			line, self._buffer = self._buffer.split("\n", 1)
			if line.strip():
				_append_check_log(self.check_id, line, self.level)
		return len(s)

	def flush(self) -> None:
		if self.original_stream:
			self.original_stream.flush()
		if self._buffer.strip():
			_append_check_log(self.check_id, self._buffer, self.level)
		self._buffer = ""


def _is_within_workspace(path: Path) -> bool:
	root = ROOT_DIR.resolve()
	target = path.resolve()
	return root == target or root in target.parents


def _dir_size_bytes(path: Path) -> int:
	if not path.exists() or not path.is_dir():
		return 0
	total = 0
	for node in path.rglob("*"):
		if node.is_file():
			total += node.stat().st_size
	return total


def _format_size(num_bytes: int) -> str:
	units = ["B", "KB", "MB", "GB", "TB"]
	size = float(max(0, num_bytes))
	for unit in units:
		if size < 1024 or unit == units[-1]:
			return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
		size /= 1024


def _build_model_check_result() -> Dict[str, Any]:
	expected_device, expected_dtype = whisper._get_optimal_device()
	expected_model_id, expected_model_name = whisper._select_model()

	actual_device = whisper._DEVICE
	actual_dtype = whisper._TORCH_DTYPE
	actual_model_id = whisper._MODEL_ID
	actual_model_name = whisper._MODEL_NAME
	loaded = whisper._pipe is not None

	items = [
		{
			"key": "device",
			"name": "设备选择",
			"ok": actual_device == expected_device,
			"expected": expected_device,
			"actual": actual_device,
		},
		{
			"key": "dtype",
			"name": "参数精度(dtype)",
			"ok": str(actual_dtype) == str(expected_dtype),
			"expected": str(expected_dtype),
			"actual": str(actual_dtype),
		},
		{
			"key": "model",
			"name": "模型选择",
			"ok": actual_model_id == expected_model_id and actual_model_name == expected_model_name,
			"expected": f"{expected_model_name} ({expected_model_id})",
			"actual": f"{actual_model_name} ({actual_model_id})",
		},
		{
			"key": "pipeline",
			"name": "Whisper pipeline 加载",
			"ok": loaded,
			"expected": "已加载",
			"actual": "已加载" if loaded else "未加载",
		},
	]

	return {
		"ok": all(item["ok"] for item in items),
		"items": items,
		"expected": {
			"device": expected_device,
			"dtype": str(expected_dtype),
			"model_id": expected_model_id,
			"model_name": expected_model_name,
		},
		"actual": {
			"device": actual_device,
			"dtype": str(actual_dtype),
			"model_id": actual_model_id,
			"model_name": actual_model_name,
			"pipeline_loaded": loaded,
		},
	}


def _split_transcription_txt(combined_txt_path: Path, output_dir: Path, stem: str) -> Dict[str, str]:
	if not combined_txt_path.exists():
		return {}

	content = combined_txt_path.read_text(encoding="utf-8")
	sep = "=" * 60

	full_match = re.search(
		r"(?:^|\n)" + re.escape(sep) + r"\n完整文本:\n" + re.escape(sep) + r"\n(.*?)(?:\n\n" + re.escape(sep) + r"|$)",
		content,
		re.DOTALL,
	)
	seg_match = re.search(
		r"(?:^|\n)" + re.escape(sep) + r"\n分段文本（带时间戳）:\n" + re.escape(sep) + r"\n(.*?)$",
		content,
		re.DOTALL,
	)

	full_text = full_match.group(1).strip() if full_match else content.strip()
	segments_text = seg_match.group(1).strip() if seg_match else ""

	full_txt = output_dir / f"{stem}_full_text.txt"
	seg_txt = output_dir / f"{stem}_segments_with_timestamps.txt"

	full_txt.write_text(full_text + "\n", encoding="utf-8")
	seg_txt.write_text((segments_text + "\n") if segments_text else "", encoding="utf-8")

	return {
		"full_txt": str(full_txt),
		"segments_txt": str(seg_txt),
		"txt": str(full_txt),
	}


def _build_transcription_progress(logs: List[Dict[str, Any]], status: str) -> Dict[str, Any]:
	total_parts = None
	completed_parts = 0

	step_init = False
	step_extract = False
	step_split = False
	step_transcribe = False

	for line in logs:
		msg = str(line.get("message", ""))

		if "步骤 1/3" in msg:
			step_init = True
		if "步骤 2/3" in msg:
			step_extract = True
		if "[split]" in msg:
			step_split = True
		if "步骤 3/3" in msg or "[transcribe] 转录:" in msg:
			step_transcribe = True

		m_total = re.search(r"\[split\]\s*最终切割为\s*(\d+)\s*段", msg)
		if m_total:
			total_parts = int(m_total.group(1))

		if "[split] 音频时长 <=" in msg and total_parts is None:
			total_parts = 1

		if "[transcribe] 转录:" in msg:
			completed_parts += 1

	if total_parts is None and completed_parts > 0:
		total_parts = completed_parts

	if total_parts and total_parts > 0:
		completed_parts = min(completed_parts, total_parts)
		percent = round((completed_parts / total_parts) * 100, 2)
	else:
		percent = 0.0

	def _step_state(done_flag: bool, active: bool) -> str:
		if status == "failed" and active:
			return "error"
		if done_flag:
			return "done"
		if active:
			return "running"
		return "waiting"

	steps = [
		{
			"key": "init",
			"name": "初始化",
			"status": _step_state(step_init, status == "running" and not step_init),
		},
		{
			"key": "extract",
			"name": "音频提取",
			"status": _step_state(step_extract, status == "running" and step_init and not step_extract),
		},
		{
			"key": "split",
			"name": "音频切割",
			"status": _step_state(step_split, status == "running" and step_extract and not step_split),
		},
		{
			"key": "transcribe",
			"name": "分段转录",
			"status": (
				"error" if status == "failed" and step_transcribe else
				"done" if (status == "completed" or (total_parts and completed_parts >= total_parts)) else
				"running" if step_transcribe else
				"waiting"
			),
			"detail": f"{completed_parts}/{total_parts}" if total_parts else "0/0",
		},
		{
			"key": "finish",
			"name": "结果产出",
			"status": "done" if status == "completed" else ("error" if status == "failed" else "waiting"),
		},
	]

	return {
		"total_parts": total_parts,
		"completed_parts": completed_parts,
		"percent": percent,
		"steps": steps,
	}


def _safe_job_dir(job_id: str) -> Path:
	job_dir = WEBUI_OUTPUT_DIR / job_id
	job_dir.mkdir(parents=True, exist_ok=True)
	return job_dir


def _run_pipeline(job_id: str, language: str) -> None:
	with _jobs_lock:
		job = _jobs.get(job_id)
		if not job:
			return
		job["status"] = "running"
		input_video = Path(job["video_path"])

	_append_log(job_id, f"任务开始：{input_video.name}")
	_append_log(job_id, "步骤 1/3：初始化音频提取器")

	out = _TeeToJob(job_id, sys.stdout, level="info")
	err = _TeeToJob(job_id, sys.stderr, level="error")

	try:
		with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
			extractor = AudioExtractor()
			_append_log(job_id, "步骤 2/3：从视频提取音频")

			job_dir = _safe_job_dir(job_id)
			audio_dir = job_dir / "audio"
			trans_dir = job_dir / "transcriptions"
			audio_dir.mkdir(parents=True, exist_ok=True)
			trans_dir.mkdir(parents=True, exist_ok=True)

			audio_out = audio_dir / f"{input_video.stem}.wav"
			extractor.extract_audio_from_video(
				video_path=input_video,
				output_path=audio_out,
				audio_format="wav",
				bitrate="192k",
				sample_rate=44100,
				channels=2,
				overwrite=True,
				verbose=True,
			)

			_append_log(job_id, "步骤 3/3：Whisper 音频切分 + 转录 + 合并")
			result = whisper.transcribe_audio(
				audio_path=audio_out,
				language=language,
				output_dir=trans_dir,
				save_txt=True,
				save_json=True,
				keep_parts=False,
			)

			files = result.get("output_files", {})
			txt_path = files.get("txt")
			json_path = files.get("json")

			split_files: Dict[str, str] = {}
			if txt_path:
				split_files = _split_transcription_txt(
					combined_txt_path=Path(txt_path),
					output_dir=trans_dir,
					stem=input_video.stem,
				)
				if split_files:
					_append_log(job_id, "已生成拆分 TXT：完整文本 + 分段时间戳文本")

			with _jobs_lock:
				job = _jobs.get(job_id)
				if not job:
					return
				job["status"] = "completed"
				job["result"] = {
					"txt": split_files.get("txt") or txt_path,
					"full_txt": split_files.get("full_txt"),
					"segments_txt": split_files.get("segments_txt"),
					"json": json_path,
				}
				job["completed_at"] = datetime.now().isoformat()

		_append_log(job_id, "任务完成，可下载 TXT / JSON 产物。", "success")
	except Exception as exc:
		tb = traceback.format_exc()
		_append_log(job_id, f"任务失败：{exc}", "error")
		_append_log(job_id, tb, "error")
		with _jobs_lock:
			job = _jobs.get(job_id)
			if job:
				job["status"] = "failed"
				job["error"] = str(exc)


@app.errorhandler(413)
def _handle_file_too_large(_error):
	return jsonify({"ok": False, "error": "上传文件超过 2GB 限制"}), 413


@app.route("/")
def index():
	return render_template("index.html")


@app.get("/api/check/model")
def check_model():
	return jsonify(_build_model_check_result())


def _run_model_check(check_id: str) -> None:
	with _checks_lock:
		check = _model_checks.get(check_id)
		if not check:
			return
		check["status"] = "running"

	_append_check_log(check_id, "开始模型检测，正在检查设备与模型配置")

	out = _TeeToCheck(check_id, sys.stdout, level="info")
	err = _TeeToCheck(check_id, sys.stderr, level="error")

	try:
		with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
			if whisper._pipe is None:
				_append_check_log(check_id, "pipeline 未加载，准备下载/加载模型（若本地不存在）")
				whisper._load_model(debug=False, debug_breakpoint=False)
			else:
				_append_check_log(check_id, "pipeline 已加载，跳过模型下载")

		result = _build_model_check_result()
		with _checks_lock:
			check = _model_checks.get(check_id)
			if not check:
				return
			check["status"] = "completed"
			check["result"] = result
			check["error"] = None
		_append_check_log(check_id, "模型检测完成", "success")
	except Exception as exc:
		with _checks_lock:
			check = _model_checks.get(check_id)
			if check:
				check["status"] = "failed"
				check["error"] = str(exc)
				check["result"] = _build_model_check_result()
		_append_check_log(check_id, f"模型检测失败：{exc}", "error")
		_append_check_log(check_id, traceback.format_exc(), "error")


@app.post("/api/check/model/start")
def start_model_check():
	check_id = uuid.uuid4().hex[:12]
	with _checks_lock:
		_model_checks[check_id] = {
			"check_id": check_id,
			"status": "queued",
			"logs": [],
			"result": None,
			"error": None,
			"created_at": datetime.now().isoformat(),
		}

	thread = threading.Thread(target=_run_model_check, args=(check_id,), daemon=True)
	thread.start()
	return jsonify({"ok": True, "check_id": check_id})


@app.get("/api/check/model/status/<check_id>")
def get_model_check_status(check_id: str):
	since_raw = request.args.get("since", "0")
	try:
		since = max(0, int(since_raw))
	except ValueError:
		since = 0

	with _checks_lock:
		check = _model_checks.get(check_id)
		if not check:
			return jsonify({"ok": False, "error": "检测任务不存在"}), 404
		logs = check.get("logs", [])
		new_logs = logs[since:]

		return jsonify(
			{
				"ok": True,
				"check_id": check_id,
				"status": check.get("status"),
				"logs": new_logs,
				"next_cursor": len(logs),
				"result": check.get("result"),
				"error": check.get("error"),
			}
		)


@app.get("/api/check/env")
def check_env():
	ffmpeg_path = shutil.which("ffmpeg")
	ffprobe_path = shutil.which("ffprobe")

	ffmpeg_version = None
	ffprobe_version = None

	if ffmpeg_path:
		try:
			proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
			if proc.returncode == 0 and proc.stdout:
				ffmpeg_version = proc.stdout.splitlines()[0]
		except Exception:
			pass

	if ffprobe_path:
		try:
			proc = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True, timeout=5)
			if proc.returncode == 0 and proc.stdout:
				ffprobe_version = proc.stdout.splitlines()[0]
		except Exception:
			pass

	libs = ["torch", "transformers", "librosa", "soundfile", "modelscope", "flask"]
	python_libs = {name: importlib.util.find_spec(name) is not None for name in libs}

	try:
		AudioExtractor()
		extractor_ready = True
		extractor_error = None
	except Exception as exc:
		extractor_ready = False
		extractor_error = str(exc)

	items: List[Dict[str, Any]] = [
		{
			"name": "ffmpeg",
			"ok": bool(ffmpeg_path),
			"detail": ffmpeg_version or ffmpeg_path or "未安装或不可执行",
		},
		{
			"name": "ffprobe",
			"ok": bool(ffprobe_path),
			"detail": ffprobe_version or ffprobe_path or "未安装或不可执行",
		},
		{
			"name": "AudioExtractor",
			"ok": extractor_ready,
			"detail": "可用" if extractor_ready else extractor_error,
		},
	]

	missing_libs = [name for name, ready in python_libs.items() if not ready]
	if missing_libs:
		py_lib_detail = "缺失: " + ", ".join(missing_libs)
	else:
		py_lib_detail = "全部已安装"

	items.append(
		{
			"name": "Python 库依赖",
			"ok": len(missing_libs) == 0,
			"detail": py_lib_detail,
		}
	)

	return jsonify(
		{
			"ok": all(item["ok"] for item in items),
			"items": items,
			"ffmpeg": {
				"path": ffmpeg_path,
				"version": ffmpeg_version,
			},
			"ffprobe": {
				"path": ffprobe_path,
				"version": ffprobe_version,
			},
			"python_libs": python_libs,
			"audio_extractor": {
				"ready": extractor_ready,
				"error": extractor_error,
			},
		}
	)


@app.get("/api/temp/usage")
def temp_usage():
	dirs = [UPLOAD_DIR, WEBUI_OUTPUT_DIR]
	details = []
	total = 0
	for p in dirs:
		if not _is_within_workspace(p):
			continue
		size = _dir_size_bytes(p)
		total += size
		details.append(
			{
				"path": str(p.relative_to(ROOT_DIR)),
				"bytes": size,
				"human": _format_size(size),
			}
		)

	return jsonify(
		{
			"ok": True,
			"total_bytes": total,
			"total_human": _format_size(total),
			"details": details,
		}
	)


@app.post("/api/temp/clear")
def clear_temp_files():
	targets = [UPLOAD_DIR, WEBUI_OUTPUT_DIR]
	removed = 0

	for base in targets:
		if not _is_within_workspace(base):
			return jsonify({"ok": False, "error": f"非法路径，已拒绝: {base}"}), 400

		base.mkdir(parents=True, exist_ok=True)
		for child in base.iterdir():
			if not _is_within_workspace(child):
				continue
			if child.is_file() or child.is_symlink():
				child.unlink(missing_ok=True)
				removed += 1
			elif child.is_dir():
				shutil.rmtree(child, ignore_errors=True)
				removed += 1

	with _jobs_lock:
		_jobs.clear()

	return jsonify({"ok": True, "removed_entries": removed})


@app.post("/api/upload")
def upload_video():
	if "file" not in request.files:
		return jsonify({"ok": False, "error": "缺少上传文件字段 file"}), 400

	file = request.files["file"]
	if not file or not file.filename:
		return jsonify({"ok": False, "error": "请选择视频文件"}), 400

	filename = secure_filename(file.filename)
	ext = Path(filename).suffix.lower()
	if ext not in AudioExtractor.SUPPORTED_FORMATS:
		return jsonify(
			{
				"ok": False,
				"error": f"不支持的视频格式: {ext}",
				"supported": sorted(AudioExtractor.SUPPORTED_FORMATS),
			}
		), 400

	job_id = uuid.uuid4().hex[:12]
	save_path = UPLOAD_DIR / f"{job_id}_{filename}"
	file.save(save_path)

	with _jobs_lock:
		_jobs[job_id] = {
			"job_id": job_id,
			"status": "uploaded",
			"video_path": str(save_path),
			"video_name": filename,
			"created_at": datetime.now().isoformat(),
			"logs": [
				{
					"ts": _now_str(),
					"level": "info",
					"message": f"上传完成：{filename}",
				}
			],
			"result": {},
			"error": None,
		}

	return jsonify({"ok": True, "job_id": job_id, "filename": filename})


@app.post("/api/start")
def start_job():
	data = request.get_json(silent=True) or {}
	job_id = (data.get("job_id") or "").strip()
	language = (data.get("language") or "zh").strip() or "zh"

	if not job_id:
		return jsonify({"ok": False, "error": "缺少 job_id"}), 400

	with _jobs_lock:
		job = _jobs.get(job_id)
		if not job:
			return jsonify({"ok": False, "error": "任务不存在"}), 404
		if job["status"] not in ("uploaded", "failed"):
			return jsonify({"ok": False, "error": f"当前状态不允许启动: {job['status']}"}), 400
		job["status"] = "queued"
		job["error"] = None

	thread = threading.Thread(target=_run_pipeline, args=(job_id, language), daemon=True)
	thread.start()

	return jsonify({"ok": True, "job_id": job_id, "status": "queued"})


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
	with _jobs_lock:
		job = _jobs.get(job_id)
		if not job:
			return jsonify({"ok": False, "error": "任务不存在"}), 404
		return jsonify(
			{
				"ok": True,
				"job": {
					"job_id": job["job_id"],
					"status": job["status"],
					"video_name": job["video_name"],
					"result": job.get("result", {}),
					"error": job.get("error"),
				},
			}
		)


@app.get("/api/jobs/<job_id>/logs")
def get_logs(job_id: str):
	since_raw = request.args.get("since", "0")
	try:
		since = max(0, int(since_raw))
	except ValueError:
		since = 0

	with _jobs_lock:
		job = _jobs.get(job_id)
		if not job:
			return jsonify({"ok": False, "error": "任务不存在"}), 404

		logs: List[Dict[str, Any]] = job.get("logs", [])
		new_logs = logs[since:]
		next_cursor = len(logs)
		progress = _build_transcription_progress(logs, job["status"])

		return jsonify(
			{
				"ok": True,
				"logs": new_logs,
				"next_cursor": next_cursor,
				"status": job["status"],
				"progress": progress,
				"error": job.get("error"),
				"result": job.get("result", {}),
			}
		)


@app.get("/api/download/<job_id>/<kind>")
def download_result(job_id: str, kind: str):
	if kind not in ("txt", "json", "full_txt", "segments_txt"):
		return jsonify({"ok": False, "error": "仅支持 txt/json/full_txt/segments_txt"}), 400

	with _jobs_lock:
		job = _jobs.get(job_id)
		if not job:
			return jsonify({"ok": False, "error": "任务不存在"}), 404
		file_path = (job.get("result") or {}).get(kind)

	if not file_path:
		return jsonify({"ok": False, "error": f"{kind} 结果尚不可用"}), 404

	p = Path(file_path)
	if not p.exists():
		return jsonify({"ok": False, "error": f"文件不存在: {p.name}"}), 404
	if not _is_within_workspace(p):
		return jsonify({"ok": False, "error": "非法下载路径"}), 400

	return send_file(p, as_attachment=True, download_name=p.name)


if __name__ == "__main__":
	host = os.getenv("UI_HOST", "127.0.0.1")
	port = int(os.getenv("UI_PORT", "5000"))
	debug = os.getenv("UI_DEBUG", "0") == "1"
	app.run(host=host, port=port, debug=debug, threaded=True)
