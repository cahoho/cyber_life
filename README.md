# welcome to CyberLife

![Engine](https://img.shields.io/badge/Engine-Unity6-green?logo=unity&logoColor=white) ![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white)

## you need to know:

- inspiration comes from **DETROIT: BECOME HUMAN**
- for study and research purpose only
- developed by me(cahoho) and ai
- using Unity6000.3.28f1

## prepare:

- download [this asset](https://assetstore.u3d.cn/packages/p/cartoon-cat-70180) and move it into folder Addons
- download [this asset](https://assetstore.u3d.cn/packages/3d/environments/assetstore-package-20001213), chose urp and move it into folder Addons.
- create a file called `.env.local` in `cyber_life/Tools/perception_service` with the follow content:

```text
SEMANTIC_PROVIDER=deepseek
DEEPSEEK_API_KEY=xxx
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
SEMANTIC_TIMEOUT_S=20
SEMANTIC_REFRESH_S=60
```

- prepare [yolo models](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) into `cyber_life/Tools/perception_service/models` 

## have a try:

```powershell
cd .\cyber_life\Tools\perception_service
.\run.ps1
```

then play the scene with vital. imgs and jsons will be saved in `cyber_life/Logs/perception ` and divided by different session.

