# 鱼类检测服务（FastAPI + YOLO）

一个基于 FastAPI 的实时/离线鱼类检测后端，使用 Ultralytics YOLO 模型进行推理，支持：
- 批量图片识别并返回统计汇总
- WebSocket 视频流推理并逐条推送鱼类检测结果
- 本地 SQLite 历史记录查询与清理

适配环境：Python 3.12，linux系统（见 `pyproject.toml`）。

## 目录结构一览

```
dataset/                # 示例输出目录（统计/预测图片等）
specs/                  # 设计文档、接口说明与计划
src/                    # 后端源代码（核心）
	api/                  # 路由层
		health.py           # 健康检查
		images.py           # 批量图片识别（REST）
		history.py          # 历史查询/删除（REST）
		stream.py           # 视频流推理（WebSocket）
	lib/                  # 通用工具
		config.py           # 配置与环境变量解析
		image_io.py         # 图像编解码
		overlay.py          # 预测框叠加绘制
	models/               # Pydantic 模型（接口 Schema）
		schemas.py
	repositories/         # 数据持久化（SQLite）
		sqlite.py
	services/             # 业务服务
		inference.py        # 模型加载与推理后处理
		history.py          # 历史读写服务
		stream_session.py   # WebSocket 会话状态
	app.py                # FastAPI 应用工厂（注册路由/启动预热）
test.py                 # WebSocket 客户端示例：抽帧发送/原速播放/本地叠加
yolo_src/               # YOLO 训练产物与配置（默认权重在此）
	best_final.pt         # 默认权重（MODEL_PATH 默认指向它）
	data.yaml             # 数据集标签等
fish_history.sqlite3    # 默认 SQLite 数据库（启动时自动创建/迁移）
```

## 快速开始

1) 安装依赖（推荐使用 [uv](https://github.com/astral-sh/uv)）：

```bash
# 使用 uv
uv sync
```

2) 环境变量（可选，见下文“环境变量”章节），最重要的是模型路径：

```bash
export MODEL_PATH="yolo_src/best_final.pt"
# 可选：过滤阈值
export MIN_CONFIDENCE=0.25
```

3) 启动服务：

```bash
uvicorn src.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

4) 健康检查：

```bash
curl http://localhost:8000/health
# -> {"status":"ok"}
```

## 环境变量（配置）

来自 `src/lib/config.py`，常用项如下：

- MODEL_PATH：权重文件路径（默认 `yolo_src/best_final.pt`）
- DB_PATH：SQLite 路径（默认 `./fish_history.sqlite3`）
- DISABLE_MODEL：设置为 `1/true` 则进入模拟模式（不加载权重）
- MIN_CONFIDENCE：最小置信度阈值，后处理会丢弃低于该值的检测（默认 0.0）
- ALLOWED_LABELS / ALLOWED_LABEL_KEYWORDS：白名单（设置后仅保留匹配的类别）
- DISALLOWED_LABELS / DISALLOWED_LABEL_KEYWORDS：黑名单（未设置白名单时生效，默认排除 human/person/water/unknown/no fish 及中文同义）
- WS_TOKENS：WebSocket 鉴权映射，格式 `clientId:token,clientId2:token2`（开启后客户端必须携带 token）
- MAX_BATCH_FILES / MAX_FILE_MB / MAX_WS_SESSIONS：批量/会话限制

说明：推理调用当前使用 Ultralytics 默认参数（imgsz/conf/iou/device 等未暴露为环境变量）。若需要，我可以帮你将其暴露出来并透传给 YOLO。

## API 使用说明

### 健康检查

- GET `/health`
- 响应：`{"status":"ok"}`

### 批量图片识别（REST）

- POST `/predict/images`
- 参数：multipart/form-data，字段名 `files`（可传多张）
- 响应模型：`ImageBatchResult`

示例：

```bash
curl -X POST http://localhost:8000/predict/images \
	-F "files=@/path/img1.jpg" \
	-F "files=@/path/img2.jpg"
```

响应（部分字段）：

```json
{
	"request_id": "...",
	"results": [
		{
			"image_id": "...",
			"width": 1920,
			"height": 1080,
			"model_version": "best_final",
			"predictions": [
				{
					"label": "Yellowfin tuna",
					"confidence": 0.86,
					"detection_index": 0,
					"bbox": {"x": 100.5, "y": 200.3, "w": 300.0, "h": 180.0}
				}
			]
		}
	],
	"summary": {
		"total_images": 2,
		"total_detections": 5,
		"category_counts": {"Yellowfin tuna": 3, "Marlin": 2},
		"confidence_min": 0.52,
		"confidence_max": 0.93,
		"confidence_avg": 0.74
	}
}
```

### 实时视频流（WebSocket）

- 地址：`ws://<host>:<port>/ws/stream`
- 鉴权：
	- 若配置了 `WS_TOKENS`，客户端需以 `?token=...` 或请求头 `x-auth-token: ...` 连接
	- 未配置 `WS_TOKENS` 时可用 `?client_id=...` 指定客户端 ID（默认 `anonymous`）
- 客户端发送内容：每帧的 JPEG 二进制（建议抽帧发送）
- 服务端响应顺序（循环）：
	1) 首条文本：`{"type":"session","session_id":"...","client_id":"..."}`
	2) 文本：`{"type":"prediction", ...}`（整帧预测元数据，含 predictions、frame_index 等）
	3) 文本：`{"type":"fish_detection", ...}`（逐条检测，多条或无）

`prediction` 字段示例：

```json
{
	"type": "prediction",
	"session_id": "...",
	"client_id": "demo-1",
	"frame_index": 42,
	"image_id": "...",
	"width": 1920,
	"height": 1080,
	"model_version": "best_final",
	"latency_ms": 37,
	"predictions": [
		{"label":"Yellowfin tuna","confidence":0.86,"detection_index":0,
		 "bbox":{"x":100.5,"y":200.3,"w":300.0,"h":180.0}}
	]
}
```

`fish_detection` 字段示例：

```json
{
	"type": "fish_detection",
	"session_id": "...",
	"client_id": "demo-1",
	"frame_index": 42,
	"detection_index": 0,
	"detection_seq": 7,
	"label": "Yellowfin tuna",
	"confidence": 0.86,
	"bbox": {"x":100.5,"y":200.3,"w":300.0,"h":180.0}
}
```

项目内提供了一个简易客户端 `test.py`，支持：

```bash
uv run ./test.py --video /path/video.mp4 --out ./out --every 5 --display \
	--token YOUR_TOKEN        # 若启用 WS_TOKENS
# 或 --client-id demo-1     # 未启用 WS_TOKENS 时可显式指定
# 可选：--save-annotated    # 保存服务端返回的标注帧
```

### 历史记录（REST）

- GET `/history/video?client_id=xxx&page=1&page_size=50` → `VideoDetectionPage`
- DELETE `/history/video?client_id=xxx` → 删除该客户端所有视频流检测记录
- GET `/history/images/{image_id}` → 单张图片的检测记录与统计 `ImageHistoryResponse`
- DELETE `/history/images/{image_id}` → 删除该图片的检测记录

> 说明：服务端会在 WebSocket 推理过程中将去重后的“鱼类检测”逐条入库；图片批量识别也会记录检测明细，便于查询与统计。
