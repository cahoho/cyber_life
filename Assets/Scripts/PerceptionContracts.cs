using System;

namespace CyberLife.Vision
{
    [Serializable]
    public sealed class PerceptionSnapshotDto
    {
        public string schema_version;
        public string session_id;
        public long source_frame_id;
        public long capture_ts_ms;
        public long result_ts_ms;
        public int infer_ms;
        public SnapshotCameraPoseDto camera_pose_at_capture;
        public PerceptionObjectDto[] objects;
        public PerceptionSceneDto scene;
        public VisionEventDto[] vision_events;
        public string[] warnings;
        public string backend;
        public PerceptionNavigationDto navigation;
    }

    [Serializable]
    public sealed class SnapshotCameraPoseDto
    {
        public float[] position;
        public float[] rotation_xyzw;
    }

    [Serializable]
    public sealed class PerceptionObjectDto
    {
        public int track_id;
        public string label;
        public string label_source;
        public float confidence;
        public float[] bbox_xyxy;
        public float[] center_uv;
        public float bbox_area_ratio;
        public float depth_m;
        public string depth_source;
        public float[] world_position;
        public float position_confidence;
        public bool moving;
        public float[] velocity_hint;
        public float track_age_s;
        public string visibility;
    }

    [Serializable]
    public sealed class PerceptionSceneDto
    {
        public float novelty;
        public bool major_change;
        public int unknown_object_count;
        public string place_hint;
        public float place_confidence;
    }

    [Serializable]
    public sealed class VisionEventDto
    {
        public string type;
        public int track_id;
        public long ts_ms;
    }

    [Serializable]
    public sealed class PerceptionNavigationDto
    {
        public string source;
        public float ground_y;
        public NavigationGridDto grid;
        public NavigationCellDto[] cells;
        public NavigationDirectionDto[] directions;
        public NavigationSummaryDto summary;
    }

    [Serializable]
    public sealed class NavigationGridDto
    {
        public float cell_size_m;
        public float forward_m;
        public float half_width_m;
        public int rows;
        public int columns;
    }

    [Serializable]
    public sealed class NavigationCellDto
    {
        public int row;
        public int column;
        public float center_right_m;
        public float center_forward_m;
        public float[] world_center;
        public string status;
        public float confidence;
        public float obstacle_height_m;
        public float[][] evidence_uv;
        public float[] evidence_u;
        public float[] evidence_v;
    }

    [Serializable]
    public sealed class NavigationDirectionDto
    {
        public float heading_deg;
        public float clearance_m;
        public string status;
    }

    [Serializable]
    public sealed class NavigationSummaryDto
    {
        public int traversable_cell_count;
        public int blocked_cell_count;
        public int unknown_cell_count;
        public float traversable_ratio;
        public float forward_clearance_m;
        public float recommended_heading_deg;
        public float recommended_clearance_m;
    }

    [Serializable]
    public sealed class PerceptionQueryResponseDto
    {
        public string question;
        public string answer;
        public bool grounded;
        public float confidence;
        public string[] evidence;
        public string source;
        public long source_frame_id;
    }
}
