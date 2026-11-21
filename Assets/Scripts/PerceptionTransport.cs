using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace CyberLife.Vision
{
    [DisallowMultipleComponent]
    public sealed class PerceptionTransport : MonoBehaviour
    {
        public enum LinkState
        {
            Connecting,
            Connected,
            Offline
        }

        [Header("服务")]
        [SerializeField] string endpoint = "http://127.0.0.1:8710/v1/frames";
        [SerializeField] float requestTimeoutSeconds = 8f;
        [SerializeField, Range(1, 4)] int maxInflightRequests = 2;

        [Header("新结果门限")]
        [SerializeField] float maxResultAgeSeconds = 0.75f;
        [SerializeField] float maxCameraMoveMeters = 0.30f;
        [SerializeField] float maxCameraRotateDegrees = 15f;

        [Header("场景问答")]
        [SerializeField] bool enableQuestionPanel = true;
        [SerializeField] string questionDraft = "场景里有没有桌子？";
        [SerializeField] float queryTimeoutSeconds = 20f;

        readonly Queue<float> rttHistory = new Queue<float>();
        int requestInflight;
        int sentFrames;
        int droppedFrames;
        int staleResults;
        int failedRequests;
        float lastFailureLogTime;

        Camera eyeCamera;
        Texture2D whiteTexture;
        GUIStyle overlayStyle;
        GUIStyle boxStyle;
        GUIStyle queryTextFieldStyle;
        GUIStyle answerStyle;
        bool questionPanelOpen;
        bool questionFocusRequested;
        bool queryPending;
        PerceptionQueryResponseDto latestAnswer;
        string queryError = string.Empty;

        public LinkState State { get; private set; } = LinkState.Connecting;
        public PerceptionSnapshotDto LatestSnapshot { get; private set; }
        public bool LatestSnapshotIsStale { get; private set; }
        public string LastError { get; private set; } = string.Empty;
        public float LastRttMs { get; private set; }
        public float RttP50Ms { get; private set; }
        public float RttP95Ms { get; private set; }
        public int SentFrames => sentFrames;
        public int DroppedFrames => droppedFrames;
        public int StaleResults => staleResults;
        public int FailedRequests => failedRequests;

        string ServiceRoot
        {
            get
            {
                const string suffix = "/v1/frames";
                return endpoint.EndsWith(suffix, StringComparison.Ordinal)
                    ? endpoint.Substring(0, endpoint.Length - suffix.Length)
                    : endpoint;
            }
        }

        void Update()
        {
            if (!enableQuestionPanel)
            {
                return;
            }
            if (Input.GetKeyDown(KeyCode.Q))
            {
                questionPanelOpen = !questionPanelOpen;
                questionFocusRequested = questionPanelOpen;
            }
            if (questionPanelOpen && Input.GetKeyDown(KeyCode.Escape))
            {
                questionPanelOpen = false;
            }
        }

        public void Configure(string serviceEndpoint, Camera cameraReference)
        {
            if (!string.IsNullOrWhiteSpace(serviceEndpoint))
            {
                endpoint = serviceEndpoint.TrimEnd('/');
                if (!endpoint.EndsWith("/v1/frames", StringComparison.Ordinal))
                {
                    endpoint += "/v1/frames";
                }
            }

            eyeCamera = cameraReference != null ? cameraReference : eyeCamera;
        }

        public void SendFrame(
            long frameId,
            long captureTsMs,
            CameraPose capturePose,
            byte[] jpeg,
            NavigationSampleSet navigation)
        {
            if (jpeg == null || jpeg.Length == 0)
            {
                LastError = "TRANSPORT_EMPTY_JPEG";
                return;
            }
            if (jpeg.Length > 400_000)
            {
                droppedFrames++;
                LastError = $"TRANSPORT_OVERSIZE: {jpeg.Length} bytes";
                LogFailure(LastError);
                return;
            }
            if (requestInflight >= maxInflightRequests)
            {
                droppedFrames++;
                LastError = "TRANSPORT_BACKPRESSURE";
                return;
            }

            string sessionId = SessionIdFromFrame(frameId, captureTsMs);
            var data = new FrameEnvelopeData
            {
                SessionId = sessionId,
                FrameId = frameId,
                CaptureTsMs = captureTsMs,
                Camera = capturePose,
                RequestFast = true,
                RequestSlow = false,
                Navigation = navigation
            };
            byte[] metadata = Encoding.UTF8.GetBytes(FrameEnvelopeJson.ToJson(data));
            requestInflight++;
            sentFrames++;
            StartCoroutine(SendFrameRoutine(frameId, captureTsMs, metadata, jpeg));
        }

        IEnumerator SendFrameRoutine(long frameId, long captureTsMs, byte[] metadata, byte[] jpeg)
        {
            var form = new WWWForm();
            form.AddBinaryData("metadata", metadata, "metadata.json", "application/json");
            form.AddBinaryData("image", jpeg, "frame.jpg", "image/jpeg");

            using (UnityWebRequest request = UnityWebRequest.Post(endpoint, form))
            {
                request.timeout = Mathf.CeilToInt(requestTimeoutSeconds);
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    HandleRequestFailure(request.error, request.responseCode);
                }
                else
                {
                    HandleResponse(frameId, captureTsMs, request.downloadHandler.text);
                }
            }
            requestInflight = Mathf.Max(0, requestInflight - 1);
        }

        public void AskScene(string question)
        {
            if (queryPending || string.IsNullOrWhiteSpace(question))
            {
                return;
            }
            StartCoroutine(AskSceneRoutine(question.Trim()));
        }

        IEnumerator AskSceneRoutine(string question)
        {
            queryPending = true;
            queryError = string.Empty;
            var payload = new QueryRequestDto { question = question };
            string json = JsonUtility.ToJson(payload);
            byte[] body = Encoding.UTF8.GetBytes(json);
            using (var request = new UnityWebRequest(
                       ServiceRoot + "/v1/query",
                       UnityWebRequest.kHttpVerbPOST,
                       new DownloadHandlerBuffer(),
                       new UploadHandlerRaw(body)))
            {
                request.SetRequestHeader("Content-Type", "application/json; charset=utf-8");
                request.timeout = Mathf.CeilToInt(queryTimeoutSeconds);
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    queryError = $"QUERY_HTTP_{request.responseCode}: {request.error}";
                }
                else
                {
                    try
                    {
                        latestAnswer = JsonUtility.FromJson<PerceptionQueryResponseDto>(
                            request.downloadHandler.text);
                    }
                    catch (Exception error)
                    {
                        queryError = $"QUERY_BAD_OUTPUT: {error.Message}";
                    }
                }
            }
            queryPending = false;
        }

        void HandleResponse(long sentFrameId, long sentCaptureTsMs, string json)
        {
            PerceptionSnapshotDto snapshot;
            try
            {
                snapshot = JsonUtility.FromJson<PerceptionSnapshotDto>(json);
            }
            catch (Exception error)
            {
                failedRequests++;
                LastError = $"PERCEPTION_BAD_OUTPUT: {error.Message}";
                State = LinkState.Offline;
                LogFailure(LastError);
                return;
            }

            if (snapshot == null || snapshot.source_frame_id != sentFrameId ||
                string.IsNullOrWhiteSpace(snapshot.session_id))
            {
                failedRequests++;
                LastError = "PERCEPTION_RESPONSE_MISMATCH";
                State = LinkState.Offline;
                LogFailure(LastError);
                return;
            }

            State = LinkState.Connected;
            LastError = string.Empty;
            long nowMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            LastRttMs = Mathf.Max(0f, nowMs - sentCaptureTsMs);
            PushRtt(LastRttMs);

            float ageSeconds = (nowMs - sentCaptureTsMs) / 1000f;
            float moveDistance = 0f;
            float rotationDegrees = 0f;
            if (eyeCamera != null && snapshot.camera_pose_at_capture != null &&
                snapshot.camera_pose_at_capture.position != null &&
                snapshot.camera_pose_at_capture.position.Length == 3 &&
                snapshot.camera_pose_at_capture.rotation_xyzw != null &&
                snapshot.camera_pose_at_capture.rotation_xyzw.Length == 4)
            {
                var p = snapshot.camera_pose_at_capture.position;
                var capturePosition = new Vector3(p[0], p[1], p[2]);
                moveDistance = Vector3.Distance(eyeCamera.transform.position, capturePosition);

                var q = snapshot.camera_pose_at_capture.rotation_xyzw;
                var captureRotation = new Quaternion(q[0], q[1], q[2], q[3]);
                rotationDegrees = Quaternion.Angle(eyeCamera.transform.rotation, captureRotation);
            }

            bool stale = ageSeconds > maxResultAgeSeconds ||
                         moveDistance > maxCameraMoveMeters ||
                         rotationDegrees > maxCameraRotateDegrees;
            if (stale)
            {
                staleResults++;
                LatestSnapshotIsStale = true;
                LastError =
                    $"STALE_RESULT age={ageSeconds:F2}s move={moveDistance:F2}m rot={rotationDegrees:F1}deg";
                return;
            }

            LatestSnapshot = snapshot;
            LatestSnapshotIsStale = false;
        }

        void HandleRequestFailure(string message, long responseCode)
        {
            failedRequests++;
            State = LinkState.Offline;
            LastError = responseCode > 0
                ? $"TRANSPORT_HTTP_{responseCode}: {message}"
                : $"TRANSPORT_OFFLINE: {message}";
            LogFailure(LastError);
        }

        void LogFailure(string message)
        {
            if (Time.unscaledTime - lastFailureLogTime < 5f)
            {
                return;
            }
            lastFailureLogTime = Time.unscaledTime;
            Debug.LogWarning($"[PerceptionTransport] {message}");
        }

        string SessionIdFromFrame(long frameId, long captureTsMs)
        {
            if (EyeCameraSensor.Active != null && !string.IsNullOrEmpty(EyeCameraSensor.Active.SessionId))
            {
                return EyeCameraSensor.Active.SessionId;
            }
            return $"sess_{captureTsMs}_{frameId}";
        }

        void PushRtt(float rttMs)
        {
            rttHistory.Enqueue(rttMs);
            while (rttHistory.Count > 120)
            {
                rttHistory.Dequeue();
            }
            var values = new List<float>(rttHistory);
            values.Sort();
            RttP50Ms = Percentile(values, 0.50f);
            RttP95Ms = Percentile(values, 0.95f);
        }

        static float Percentile(List<float> sortedValues, float percentile)
        {
            if (sortedValues.Count == 0)
            {
                return 0f;
            }
            int index = Mathf.Clamp(
                Mathf.RoundToInt((sortedValues.Count - 1) * percentile),
                0,
                sortedValues.Count - 1);
            return sortedValues[index];
        }

        void OnGUI()
        {
            EnsureStyles();
            if (Event.current.type == EventType.Repaint)
            {
                DrawObjectBoxes();
                DrawNavigationOverlay();
                DrawStatusPanel();
            }

            // IMGUI controls need Layout, Repaint and mouse/key events. Returning
            // early for non-Repaint events makes TextField/Button non-interactive.
            DrawQuestionPanel();
        }

        void DrawStatusPanel()
        {
            const float margin = 10f;
            const float panelWidth = 510f;
            const float panelHeight = 114f;
            GUI.color = new Color(0.02f, 0.04f, 0.07f, 0.82f);
            GUI.Box(new Rect(margin, margin, panelWidth, panelHeight), GUIContent.none, boxStyle);
            GUI.color = Color.white;

            Color previousColor = GUI.color;
            GUI.color = State == LinkState.Connected
                ? new Color(0.35f, 1f, 0.65f)
                : State == LinkState.Connecting
                    ? new Color(1f, 0.80f, 0.30f)
                    : new Color(1f, 0.38f, 0.32f);
            GUI.Label(
                new Rect(margin + 12f, margin + 8f, panelWidth - 24f, 18f),
                $"VISION {State}   backend={(LatestSnapshot != null ? LatestSnapshot.backend : "waiting")}",
                overlayStyle);
            GUI.color = previousColor;

            string frameText = LatestSnapshot != null
                ? $"frame={LatestSnapshot.source_frame_id} infer={LatestSnapshot.infer_ms}ms " +
                  $"objects={LatestSnapshot.objects?.Length ?? 0}"
                : "frame=-- objects=--";
            GUI.Label(
                new Rect(margin + 12f, margin + 30f, panelWidth - 24f, 18f),
                $"{frameText}   RTT p50={RttP50Ms:F0}ms p95={RttP95Ms:F0}ms",
                overlayStyle);
            GUI.Label(
                new Rect(margin + 12f, margin + 51f, panelWidth - 24f, 18f),
                LatestSnapshot?.navigation?.summary != null
                    ? $"NAV clear={LatestSnapshot.navigation.summary.forward_clearance_m:F1}m " +
                      $"best={LatestSnapshot.navigation.summary.recommended_heading_deg:F0}deg " +
                      $"blocked={LatestSnapshot.navigation.summary.blocked_cell_count}"
                    : "NAV unavailable (no depth/ray samples)",
                overlayStyle);
            GUI.Label(
                new Rect(margin + 12f, margin + 72f, panelWidth - 24f, 18f),
                $"sent={sentFrames} dropped={droppedFrames} stale={staleResults} failed={failedRequests}",
                overlayStyle);
            GUI.Label(
                new Rect(margin + 12f, margin + 93f, panelWidth - 24f, 18f),
                string.IsNullOrEmpty(LastError) ? "S 切换采集  Q 场景问答" : LastError,
                overlayStyle);
        }

        void DrawObjectBoxes()
        {
            if (LatestSnapshot?.objects == null || LatestSnapshotIsStale)
            {
                return;
            }
            if (!TryGetCameraImageRect(out Rect imageRect))
            {
                GUI.Label(
                    new Rect(10f, 112f, 510f, 20f),
                    "VISION overlay: 未找到 OutputRT RawImage，使用全屏回退映射",
                    overlayStyle);
                imageRect = new Rect(0f, 0f, Screen.width, Screen.height);
            }

            foreach (PerceptionObjectDto item in LatestSnapshot.objects)
            {
                if (item?.bbox_xyxy == null || item.bbox_xyxy.Length != 4)
                {
                    continue;
                }

                float normalizedX = item.bbox_xyxy[0] / LatestSnapshotCameraWidth();
                float normalizedY = item.bbox_xyxy[1] / LatestSnapshotCameraHeight();
                float normalizedWidth =
                    (item.bbox_xyxy[2] - item.bbox_xyxy[0]) / LatestSnapshotCameraWidth();
                float normalizedHeight =
                    (item.bbox_xyxy[3] - item.bbox_xyxy[1]) / LatestSnapshotCameraHeight();
                Rect rect = new Rect(
                    imageRect.x + normalizedX * imageRect.width,
                    imageRect.y + normalizedY * imageRect.height,
                    normalizedWidth * imageRect.width,
                    normalizedHeight * imageRect.height);

                Color color = item.label == "other"
                    ? new Color(1f, 0.55f, 0.2f, 0.95f)
                    : new Color(0.2f, 0.95f, 0.75f, 0.95f);
                DrawRect(rect, color, 2f);
                string label =
                    $"{item.label} #{item.track_id} {item.confidence:F2}" +
                    (item.moving ? "  MOVE" : string.Empty);
                var labelRect = new Rect(rect.x, Mathf.Max(2f, rect.y - 21f), 220f, 20f);
                GUI.color = new Color(0.02f, 0.04f, 0.07f, 0.85f);
                GUI.Box(labelRect, GUIContent.none, boxStyle);
                GUI.color = color;
                GUI.Label(new Rect(labelRect.x + 4f, labelRect.y + 2f, labelRect.width, 18f), label, overlayStyle);
                GUI.color = Color.white;
            }
        }

        void DrawNavigationOverlay()
        {
            PerceptionNavigationDto navigation = LatestSnapshot?.navigation;
            if (LatestSnapshotIsStale || navigation?.cells == null ||
                !TryGetCameraImageRect(out Rect imageRect))
            {
                return;
            }

            foreach (NavigationCellDto cell in navigation.cells)
            {
                if (cell?.evidence_u == null || cell.evidence_v == null ||
                    cell.evidence_u.Length == 0 ||
                    cell.evidence_u.Length != cell.evidence_v.Length ||
                    cell.status == "unknown")
                {
                    continue;
                }
                Color color = cell.status == "traversable"
                    ? new Color(0.2f, 1f, 0.45f, 0.9f)
                    : new Color(1f, 0.25f, 0.18f, 0.95f);
                float size = cell.status == "blocked" ? 9f : 6f;
                GUI.color = color;
                for (int i = 0; i < cell.evidence_u.Length; i++)
                {
                    GUI.DrawTexture(
                        new Rect(
                            imageRect.x + cell.evidence_u[i] * imageRect.width - size * 0.5f,
                            imageRect.y + cell.evidence_v[i] * imageRect.height - size * 0.5f,
                            size,
                            size),
                        whiteTexture);
                }
            }
            GUI.color = Color.white;
        }

        void DrawQuestionPanel()
        {
            if (!questionPanelOpen)
            {
                return;
            }
            const float width = 720f;
            const float height = 150f;
            float x = Mathf.Max(10f, (Screen.width - width) * 0.5f);
            float y = Mathf.Max(138f, Screen.height - height - 12f);
            GUI.color = new Color(0.02f, 0.04f, 0.07f, 0.92f);
            GUI.Box(new Rect(x, y, width, height), GUIContent.none, boxStyle);
            GUI.color = Color.white;

            GUI.SetNextControlName("PerceptionQuestion");
            questionDraft = GUI.TextField(
                new Rect(x + 10f, y + 9f, width - 90f, 26f),
                questionDraft ?? string.Empty,
                2000,
                queryTextFieldStyle);
            bool submit = GUI.Button(new Rect(x + width - 72f, y + 9f, 62f, 26f), "询问");
            if (questionFocusRequested)
            {
                GUI.FocusControl("PerceptionQuestion");
                questionFocusRequested = false;
            }

            Event current = Event.current;
            if (current.type == EventType.KeyDown &&
                (current.keyCode == KeyCode.Return || current.keyCode == KeyCode.KeypadEnter) &&
                GUI.GetNameOfFocusedControl() == "PerceptionQuestion")
            {
                submit = true;
                current.Use();
            }
            if (submit && !queryPending)
            {
                AskScene(questionDraft);
            }

            string answer = queryPending
                ? "正在请求场景理解…"
                : !string.IsNullOrEmpty(queryError)
                    ? queryError
                    : latestAnswer != null
                        ? $"[{latestAnswer.source} conf={latestAnswer.confidence:F2}] " +
                          latestAnswer.answer
                    : "输入问题后点击“询问”或按 Enter。DeepSeek 回复会显示在这里。";
            GUI.Label(new Rect(x + 10f, y + 43f, width - 20f, height - 53f), answer, answerStyle);
        }

        bool TryGetCameraImageRect(out Rect imageRect)
        {
            imageRect = default;
            Texture sourceTexture = eyeCamera != null ? eyeCamera.targetTexture : null;
            if (sourceTexture == null)
            {
                return false;
            }

            RawImage[] rawImages = FindObjectsOfType<RawImage>(includeInactive: true);
            foreach (RawImage rawImage in rawImages)
            {
                if (rawImage == null || rawImage.texture != sourceTexture ||
                    !rawImage.isActiveAndEnabled)
                {
                    continue;
                }

                var rectTransform = rawImage.rectTransform;
                Vector3[] corners = new Vector3[4];
                rectTransform.GetWorldCorners(corners);
                Vector2 bottomLeft = RectTransformUtility.WorldToScreenPoint(null, corners[0]);
                Vector2 topLeft = RectTransformUtility.WorldToScreenPoint(null, corners[1]);
                Vector2 topRight = RectTransformUtility.WorldToScreenPoint(null, corners[2]);
                Vector2 bottomRight = RectTransformUtility.WorldToScreenPoint(null, corners[3]);

                float left = Mathf.Min(bottomLeft.x, topLeft.x, topRight.x, bottomRight.x);
                float right = Mathf.Max(bottomLeft.x, topLeft.x, topRight.x, bottomRight.x);
                float bottom = Mathf.Min(bottomLeft.y, topLeft.y, topRight.y, bottomRight.y);
                float top = Mathf.Max(bottomLeft.y, topLeft.y, topRight.y, bottomRight.y);
                if (right - left < 1f || top - bottom < 1f)
                {
                    continue;
                }

                // UGUI screen space is bottom-left based; IMGUI is top-left based.
                imageRect = new Rect(left, Screen.height - top, right - left, top - bottom);
                return true;
            }
            return false;
        }

        static int LatestSnapshotCameraWidth()
        {
            return PerceptionOverlayConfig.CameraWidth;
        }

        static int LatestSnapshotCameraHeight()
        {
            return PerceptionOverlayConfig.CameraHeight;
        }

        void DrawRect(Rect rect, Color color, float thickness)
        {
            GUI.color = color;
            GUI.DrawTexture(new Rect(rect.x, rect.y, rect.width, thickness), whiteTexture);
            GUI.DrawTexture(new Rect(rect.x, rect.yMax - thickness, rect.width, thickness), whiteTexture);
            GUI.DrawTexture(new Rect(rect.x, rect.y, thickness, rect.height), whiteTexture);
            GUI.DrawTexture(new Rect(rect.xMax - thickness, rect.y, thickness, rect.height), whiteTexture);
            GUI.color = Color.white;
        }

        void EnsureStyles()
        {
            if (whiteTexture == null)
            {
                whiteTexture = Texture2D.whiteTexture;
            }
            if (overlayStyle == null)
            {
                overlayStyle = new GUIStyle(GUI.skin.label)
                {
                    fontSize = 12,
                    fontStyle = FontStyle.Bold,
                    alignment = TextAnchor.MiddleLeft,
                    clipping = TextClipping.Clip,
                    normal = { textColor = Color.white }
                };
            }
            if (boxStyle == null)
            {
                boxStyle = new GUIStyle(GUI.skin.box);
            }
            if (queryTextFieldStyle == null)
            {
                queryTextFieldStyle = new GUIStyle(GUI.skin.textField)
                {
                    fontSize = 14,
                    alignment = TextAnchor.MiddleLeft
                };
            }
            if (answerStyle == null)
            {
                answerStyle = new GUIStyle(GUI.skin.label)
                {
                    fontSize = 14,
                    fontStyle = FontStyle.Normal,
                    alignment = TextAnchor.UpperLeft,
                    wordWrap = true,
                    clipping = TextClipping.Clip,
                    normal = { textColor = Color.white }
                };
            }
        }

        [Serializable]
        sealed class QueryRequestDto
        {
            public string question;
        }
    }

    internal static class PerceptionOverlayConfig
    {
        public static int CameraWidth { get; set; } = 640;
        public static int CameraHeight { get; set; } = 360;
    }
}
