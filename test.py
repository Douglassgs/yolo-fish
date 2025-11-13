from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncIterator, Tuple

import cv2  # type: ignore[import-untyped]
import websockets  # type: ignore[import-untyped]
from websockets.exceptions import ConnectionClosed  # type: ignore[import-untyped]
import numpy as np 
from PIL import Image
from src.lib.overlay import draw_overlays
from urllib.parse import urlencode


async def iter_video_frames(video_path: Path, every: int) -> AsyncIterator[Tuple[int, "cv2.Mat"]]:
    """
    异步迭代视频帧：每隔 `every` 帧取一帧。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every == 0:
                yield idx, frame
            idx += 1
            # 轻微让出事件循环
            await asyncio.sleep(0)
    finally:
        cap.release()


async def iter_all_frames(video_path: Path) -> AsyncIterator[Tuple[int, "cv2.Mat"]]:
    """异步逐帧迭代整个视频（不抽帧）。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
            await asyncio.sleep(0)
    finally:
        cap.release()


def get_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 25.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    try:
        fps = float(fps)
    except Exception:
        fps = 25.0
    if fps <= 1e-3:
        fps = 25.0
    return fps


def encode_jpeg(frame: "cv2.Mat", quality: int = 90) -> bytes:
    """
    将 OpenCV BGR 帧编码为 JPEG 字节流。
    """
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", frame, params)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


async def send_video_over_ws(
    uri: str,
    video_path: Path,
    out_dir: Path,
    every: int = 3,
    quality: int = 90,
    limit: int | None = None,
    display: bool = False,
    window_name: str = "YOLO Stream",
    token: str | None = None,
    client_id: str | None = None,
    save_annotated: bool = False,
) -> None:
    """
    通过 WebSocket 发送视频帧，并保存后端返回的预测与标注图。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = out_dir / "predictions.ndjson"
    det_ndjson_path = out_dir / "detections.ndjson"
    ann_dir = out_dir / "pred_images"
    if save_annotated:
        ann_dir.mkdir(parents=True, exist_ok=True)

    user_requested_stop = False  # 新增

    try:
        # 认证与客户端标识：优先 token，其次 client_id
        query: list[tuple[str, str]] = []
        if token:
            query.append(("token", token))
        elif client_id:
            query.append(("client_id", client_id))
        if query:
            q = urlencode(query)
            final_uri = f"{uri}?{q}"
        else:
            final_uri = uri

        async with websockets.connect(final_uri, max_size=32 * 1024 * 1024) as ws:
            # 接收会话信息
            first = await ws.recv()
            if isinstance(first, (bytes, bytearray)):
                raise RuntimeError("期望先收到会话文本消息，但却收到了二进制数据。")
            try:
                msg = json.loads(first)
            except json.JSONDecodeError:
                raise RuntimeError(f"无法解析会话消息: {first!r}")
            if msg.get("type") != "session":
                raise RuntimeError(f"首条消息不是会话信息: {msg}")
            session_id = msg.get("session_id")
            server_client_id = msg.get("client_id")
            print(f"[ws] 会话已建立: session_id={session_id} client_id={server_client_id}")

            sent = 0
            # 最近一次预测（用于在两次预测之间持续显示）
            last_pred: dict | None = None
            # 原速播放节流参数
            fps = get_video_fps(video_path)
            frame_dt = 1.0 / fps
            loop = asyncio.get_running_loop()
            start_t = loop.time()
            shown = 0

            async def recv_loop() -> None:
                nonlocal last_pred
                while True:
                    msg_any = await ws.recv()
                    if isinstance(msg_any, (bytes, bytearray)):
                        # 可能是上一条 prediction 的标注图
                        if save_annotated and last_pred is not None:
                            try:
                                frame_index = int(last_pred.get("frame_index", 0))
                                arr = np.frombuffer(msg_any, dtype=np.uint8)
                                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if img is not None:
                                    out_path = ann_dir / f"frame_{frame_index:06d}.jpg"
                                    cv2.imwrite(str(out_path), img)
                            except Exception as e:  # noqa: BLE001
                                print(f"[ws] 保存标注图失败: {e}", file=sys.stderr)
                        continue
                    # 文本消息
                    try:
                        data = json.loads(msg_any)
                    except json.JSONDecodeError:
                        print(f"[ws] 无法解析的文本消息: {msg_any!r}", file=sys.stderr)
                        continue

                    mtype = data.get("type")
                    if mtype == "prediction":
                        last_pred = data
                        # 记录到 NDJSON
                        with ndjson_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        # 读取紧随其后的标注 JPEG（二进制），若不保存则丢弃
                        try:
                            next_any = await ws.recv()
                            if isinstance(next_any, (bytes, bytearray)):
                                if save_annotated:
                                    try:
                                        frame_index = int(last_pred.get("frame_index", 0))
                                        arr = np.frombuffer(next_any, dtype=np.uint8)
                                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                        if img is not None:
                                            out_path = ann_dir / f"frame_{frame_index:06d}.jpg"
                                            cv2.imwrite(str(out_path), img)
                                    except Exception as e:  # noqa: BLE001
                                        print(f"[ws] 保存标注图失败: {e}", file=sys.stderr)
                        except Exception as e:  # noqa: BLE001
                            print(f"[ws] 读取标注图失败: {e}", file=sys.stderr)
                    elif mtype == "fish_detection":
                        # 逐条检测结果
                        with det_ndjson_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    elif mtype == "error":
                        print(f"[ws] 后端错误: {data.get('message')}", file=sys.stderr)
                    elif mtype == "session":
                        # 重复会话信息，忽略
                        continue
                    else:
                        print(f"[ws] 未知消息类型: {data}", file=sys.stderr)

            recv_task = asyncio.create_task(recv_loop())

            async for idx, frame in iter_all_frames(video_path):
                # 可选：限制发送的帧数
                if limit is not None and sent >= limit:
                    break
                if user_requested_stop:
                    break  # 新增

                # 在抽帧节奏下发送帧（只发送，当 idx%every==0）
                if idx % every == 0:
                    jpeg = encode_jpeg(frame, quality=quality)
                    await ws.send(jpeg)
                    sent += 1

                # 基于最近一次预测在本地原始帧叠加绘制（直到下一次预测到来之前持续显示）
                cur_pred = last_pred
                preds = (cur_pred or {}).get("predictions", []) or []
                boxes = [
                    (
                        float(p.get("bbox", {}).get("x", 0.0)),
                        float(p.get("bbox", {}).get("y", 0.0)),
                        float(p.get("bbox", {}).get("w", 0.0)),
                        float(p.get("bbox", {}).get("h", 0.0)),
                    )
                    for p in preds
                ]
                labels = [str(p.get("label", "")) for p in preds]
                scores = [float(p.get("confidence", 0.0)) for p in preds]

                # BGR -> PIL RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                pil_drawn = draw_overlays(pil_img, boxes=boxes, labels=labels, scores=scores)
                # PIL RGB -> BGR for OpenCV
                shown_img = cv2.cvtColor(np.array(pil_drawn), cv2.COLOR_RGB2BGR)

                # 实时显示（如果开启），尝试按视频 FPS 原速播放
                if display:
                    target_t = start_t + shown * frame_dt
                    now = loop.time()
                    delay = target_t - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                    cv2.imshow(window_name, shown_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # q 或 Esc 退出
                        user_requested_stop = True
                    shown += 1

                # 打印状态（可选）
                if idx % max(1, every * 10) == 0:
                    print(
                        f"[ok] frame={idx} sent={sent} last_preds={len(preds)} latency={ (cur_pred or {}).get('latency_ms') }ms"
                    )

                if user_requested_stop:
                    break  # 新增

            # 正常结束后关闭
            await ws.close()
            print("[ws] 发送完成，连接已关闭。")
            # 结束接收任务
            recv_task.cancel()
            try:
                await recv_task
            except Exception:
                pass
    except ConnectionClosed as e:
        print(f"[ws] 连接关闭: code={getattr(e, 'code', None)} reason={getattr(e, 'reason', None)}")
    finally:
        if display:
            cv2.destroyAllWindows()  # 新增


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按原视频帧率播放；每隔 N 帧抽取一帧发送预测，并在两次预测之间持续显示上一次预测框。")
    p.add_argument("--video", required=True, type=Path, help="输入视频路径")
    p.add_argument("--out", required=True, type=Path, help="输出目录，用于保存标注图与 predictions.ndjson")
    p.add_argument("--every", type=int, default=3, help="每隔多少帧取一帧（默认 3）")
    p.add_argument("--quality", type=int, default=90, help="JPEG 质量（默认 90）")
    p.add_argument("--limit", type=int, default=None, help="最多发送多少张（默认无限制）")
    p.add_argument("--host", default="localhost", help="后端主机（默认 localhost）")
    p.add_argument("--port", type=int, default=8000, help="后端端口（默认 8000）")
    p.add_argument("--display", action="store_true", help="实时显示叠加预测的视频窗口")
    p.add_argument("--token", default=None, help="认证 token（若配置了 WS_TOKENS，必须提供）")
    p.add_argument("--client-id", dest="client_id", default=None, help="客户端标识（当未配置 WS_TOKENS 时可用）")
    p.add_argument("--save-annotated", action="store_true", help="保存后端返回的标注帧 JPEG")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uri = f"ws://{args.host}:{args.port}/ws/stream"
    try:
        asyncio.run(
            send_video_over_ws(
                uri=uri,
                video_path=args.video,
                out_dir=args.out,
                every=args.every,
                quality=args.quality,
                limit=args.limit,
                display=args.display,
                token=args.token,
                client_id=args.client_id,
                save_annotated=args.save_annotated,
            )
        )
    except KeyboardInterrupt:
        print("\n[ws] 已中断。")