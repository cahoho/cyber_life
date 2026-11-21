# CyberLife 视觉理解服务

这是第一阶段 `M1.2 快理解骨架` 的可运行首版。它接收 Unity 的
`FrameEnvelope` + JPEG，返回 `PerceptionSnapshot`。

## 当前能力

- YOLO 检测与 Ultralytics ByteTrack 稳定跟踪。
- 结构化 2D 快照：类别、置信度、bbox、归一化中心、面积比、运动提示。
- Unity 相机射线导航采样：把可见地面、障碍与未知区域转换为
  `camera_ground` 坐标下的 0.5 m 通行网格。
- `GET /v1/state`：查询最新相机位置、对象、地点假设与前方净空。
- `POST /v1/query`：自然语言场景问答。没有 API Key 时仍能回答物体是否存在、
  通行性、当前位置和已推断出的地点；配置 API Key 后增加开放语义问答。
- 可选 VLM 场景理解：地点类型、可见地标、场景描述和不确定性，低频异步执行，
  不阻塞检测主链路。
- `TRACK_APPEARED`、`TARGET_LOST`、`NOVEL_OBJECT`、`SCENE_CHANGED` 事件。
- 模型缺失时自动使用离线前景检测后端，传输与快照契约仍可验证。
- 未提供射线/深度时，三维位置和通行性诚实返回 `null`，不会用语言模型猜测几何。
- 单消费者顺序处理，保持跟踪器帧序；队列满时以最新帧替代旧帧。

## 启动

首次使用可下载模型权重：

```powershell
cd Tools\perception_service
.\download_model.ps1
```

下载后权重位于：

```text
Tools/perception_service/models/yolov8n.pt
```

然后运行：

```powershell
cd Tools\perception_service
.\run.ps1
```

没有模型时仍可启动：

```powershell
.\run.ps1 -Backend heuristic
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8710/health
```

查看最新状态：

```powershell
Invoke-RestMethod http://127.0.0.1:8710/v1/state
```

场景问答：

```powershell
.\ask.ps1 "场景里有没有桌子？"
.\ask.ps1 "前方可以通行吗？"
.\ask.ps1 "我现在可能在什么地方？"
```

也可以直接调用 JSON API：

```powershell
Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"question":"场景里有没有椅子？"}' `
  http://127.0.0.1:8710/v1/query
```

## 接入 DeepSeek V4.1 Flash

服务默认支持 DeepSeek Responses API。DeepSeek 官方模型标识
`deepseek-flash` 当前对应 `DeepSeek-V4.1-Flash-1017`，支持图片输入和
JSON Schema 结构化输出。

本项目已支持从 `Tools/perception_service/.env.local` 自动加载密钥。该文件已
被 `.gitignore` 排除，Key 不会写入 Unity、帧 JSON 或日志。也可以直接使用进程
环境变量：

```powershell
$env:SEMANTIC_PROVIDER = "deepseek"
$env:DEEPSEEK_API_KEY = "<your-api-key>"
$env:DEEPSEEK_MODEL = "deepseek-flash"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
.\run.ps1
```

可选配置：

```text
SEMANTIC_PROVIDER        deepseek / openai / auto
SEMANTIC_TIMEOUT_S       默认 20 秒
SEMANTIC_REFRESH_S       默认 60 秒
DEEPSEEK_MODEL           默认 deepseek-flash
DEEPSEEK_BASE_URL        默认 https://api.deepseek.com
```

Key 不配置时服务仍以本地模式完整运行。不要把 Key 提交到 Git、场景资产、
Inspector 字段或客户端程序；发布版应由单独的可信后端代理 API。

Unity 内的 `EyeCameraSensor` 会自动创建 HTTP 客户端。服务不可用时只显示离线状态，
不影响 Play Mode 或取帧。

## Unity 调试

1. 启动服务，访问 `/health` 确认 `status=ready`。
2. 打开 `Assets/Scenes/SampleScene.scene`，进入 Play Mode。
3. 采集默认自动开始；`S` 键可暂停或恢复。
4. Game 视图左上角应显示连接、RTT、帧号、对象数和 `NAV clear=...` 净空。
5. 画面中的绿色小点表示射线观测到的可通行地面，红色小点表示障碍。
6. 按 `Q` 打开问答栏，输入问题后按 Enter。
7. 停止服务后，Unity 应继续运行并把状态切到 `Offline`，不会阻塞渲染。

如果 Unity Console 提示 `TRANSPORT_OFFLINE`，先确认端口 `8710`、服务日志和
Windows 防火墙。如果显示 `STALE_RESULT`，说明结果晚于相机位姿允许窗口，
这属于预期丢弃，不会写入后续 Belief。

## 通行网格口径

Unity 的 `UnityRaycastNavigationSampler` 在每次采集时从前向相机视口发
`24 x 14` 条射线。Python 将命中点投影到相机水平前向/右向坐标，输出：

- `traversable`：接近地面的水平表面，且在允许台阶高度内。
- `blocked`：陡峭表面，或高于地面允许台阶的台面/物体。
- `unknown`：没有射线命中或采样不足，绝不能当成可通行。

`directions[]` 给出左 60° 到右 60°、每 15° 一档的净空；
`recommended_heading_deg` 只是基于当前单帧几何的建议，不是完整路径规划。
碰撞体缺失、玻璃、悬空障碍、动态物体和遮挡盲区仍是已知边界。
