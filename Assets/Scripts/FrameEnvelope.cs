///<summary>
/// 这里定义帧元数据
/// </summary>

using System;
using UnityEngine;

namespace CyberLife.Vision
{
    public struct FrameEnvelopeData
    {
        public string SessionId;
        public long FrameId;
        public long CaptureTsMs;
        public CameraPose Camera;
        public bool RequestFast;
        public bool RequestSlow;
        public NavigationSampleSet Navigation;
    }
    public static class FrameEnvelopeJson
    {
        public static string ToJson(FrameEnvelopeData d) // JsonUtility无法表达null。
        {
            string navigationJson = NavigationSampleJson.ToJson(d.Navigation);
            return FormattableString.Invariant($@"{{
    ""schema_version"": ""1.0"",
    ""session_id"": ""{d.SessionId}"",
    ""frame_id"": {d.FrameId},
    ""capture_ts_ms"": {d.CaptureTsMs},
    ""camera"": {{
        ""width"": {d.Camera.width},
        ""height"": {d.Camera.height},
        ""fov_y_deg"": {d.Camera.fovYDeg:R},
        ""near"": {d.Camera.near:R},
        ""far"": {d.Camera.far:R},
        ""position"": [
        {d.Camera.position.x:R},
        {d.Camera.position.y:R},
        {d.Camera.position.z:R}
        ],
        ""rotation_xyzw"": [
        {d.Camera.rotation.x:R},
        {d.Camera.rotation.y:R},
        {d.Camera.rotation.z:R},
        {d.Camera.rotation.w:R}
        ],
        ""unity_to_cv"": true
    }},
    ""request"": {{
        ""fast"": {(d.RequestFast ? "true" : "false")},
        ""slow"": {(d.RequestSlow ? "true" : "false")},
        ""slow_reason"": null,
        ""roi"": null
    }},
    ""navigation"": {navigationJson}
    }}");
        }
    }

    internal static class NavigationSampleJson
    {
        public static string ToJson(NavigationSampleSet samples)
        {
            if (samples == null || samples.rays == null)
            {
                return "null";
            }

            var json = new System.Text.StringBuilder(32768);
            json.Append("{\"source\":\"");
            json.Append(samples.source ?? "unity_raycast");
            json.Append("\",\"ground_y\":");
            json.Append(samples.groundY.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            json.Append(",\"max_slope_deg\":");
            json.Append(samples.maxSlopeDegrees.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            json.Append(",\"step_height_m\":");
            json.Append(samples.stepHeightMeters.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
            json.Append(",\"rays\":[");
            for (int i = 0; i < samples.rays.Length; i++)
            {
                if (i > 0)
                {
                    json.Append(',');
                }
                NavigationRaySample ray = samples.rays[i];
                json.Append("{\"pixel_uv\":[");
                AppendFloat(json, ray.pixelUv.x);
                json.Append(',');
                AppendFloat(json, ray.pixelUv.y);
                json.Append("],\"direction_camera\":[");
                AppendVector3(json, ray.directionCamera);
                json.Append("],\"hit\":");
                json.Append(ray.hit ? "true" : "false");
                json.Append(",\"hit_distance_m\":");
                if (ray.hit)
                {
                    AppendFloat(json, ray.hitDistanceMeters);
                    json.Append(",\"world_position\":[");
                    AppendVector3(json, ray.worldPosition);
                    json.Append("],\"world_normal\":[");
                    AppendVector3(json, ray.worldNormal);
                    json.Append("],\"surface\":\"");
                    json.Append(ray.surface ?? "unknown");
                    json.Append('"');
                }
                else
                {
                    json.Append("null,\"world_position\":null,\"world_normal\":null,\"surface\":\"unknown\"");
                }
                json.Append('}');
            }
            json.Append("]}");
            return json.ToString();
        }

        static void AppendVector3(System.Text.StringBuilder json, Vector3 value)
        {
            AppendFloat(json, value.x);
            json.Append(',');
            AppendFloat(json, value.y);
            json.Append(',');
            AppendFloat(json, value.z);
        }

        static void AppendFloat(System.Text.StringBuilder json, float value)
        {
            json.Append(value.ToString("R", System.Globalization.CultureInfo.InvariantCulture));
        }
    }
}
