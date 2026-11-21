using System;
using UnityEngine;

namespace CyberLife.Vision
{
    [DisallowMultipleComponent]
    public sealed class UnityRaycastNavigationSampler : MonoBehaviour
    {
        [SerializeField] Camera eyeCamera;
        [SerializeField, Range(4, 32)] int columns = 24;
        [SerializeField, Range(4, 18)] int rows = 14;
        [SerializeField, Range(1f, 20f)] float maxRangeMeters = 8f;
        [SerializeField, Range(5f, 80f)] float maxSlopeDegrees = 45f;
        [SerializeField, Range(0f, 0.5f)] float stepHeightMeters = 0.2f;
        [SerializeField] LayerMask raycastMask = ~0;
        [SerializeField, Range(0.25f, 0.9f)] float topViewport = 0.28f;
        [SerializeField, Range(0.5f, 1f)] float bottomViewport = 0.96f;

        readonly RaycastHit[] hitBuffer = new RaycastHit[32];

        public void Configure(Camera cameraReference)
        {
            if (eyeCamera == null)
            {
                eyeCamera = cameraReference;
            }
        }

        public NavigationSampleSet Sample()
        {
            if (eyeCamera == null)
            {
                return null;
            }

            float groundY = FindGroundY();
            var result = new NavigationSampleSet
            {
                source = "unity_raycast",
                groundY = groundY,
                maxSlopeDegrees = maxSlopeDegrees,
                stepHeightMeters = stepHeightMeters,
                rays = new NavigationRaySample[columns * rows]
            };

            int index = 0;
            for (int row = 0; row < rows; row++)
            {
                float v = Mathf.Lerp(
                    topViewport,
                    bottomViewport,
                    rows > 1 ? row / (float)(rows - 1) : 0f);
                for (int column = 0; column < columns; column++)
                {
                    float u = (column + 0.5f) / columns;
                    Ray ray = eyeCamera.ViewportPointToRay(new Vector3(u, 1f - v, 0f));
                    Vector3 directionCamera =
                        eyeCamera.transform.InverseTransformDirection(ray.direction).normalized;
                    var sample = new NavigationRaySample
                    {
                        pixelUv = new Vector2(u, v),
                        directionCamera = directionCamera,
                        hit = false,
                        surface = "unknown"
                    };

                    if (TryClosestHit(ray, out RaycastHit hit))
                    {
                        sample.hit = true;
                        sample.hitDistanceMeters = hit.distance;
                        sample.worldPosition = hit.point;
                        sample.worldNormal = hit.normal;
                        sample.surface = ClassifySurface(hit, groundY);
                    }
                    result.rays[index++] = sample;
                }
            }
            return result;
        }

        bool TryClosestHit(Ray ray, out RaycastHit closest)
        {
            closest = default;
            int count = Physics.RaycastNonAlloc(
                ray,
                hitBuffer,
                maxRangeMeters,
                raycastMask,
                QueryTriggerInteraction.Ignore);
            bool found = false;
            float nearest = float.PositiveInfinity;
            for (int i = 0; i < count; i++)
            {
                RaycastHit candidate = hitBuffer[i];
                if (candidate.collider == null || IsSelf(candidate.collider))
                {
                    continue;
                }
                if (candidate.distance < nearest)
                {
                    nearest = candidate.distance;
                    closest = candidate;
                    found = true;
                }
            }
            return found;
        }

        float FindGroundY()
        {
            Transform bodyRoot = eyeCamera.transform.root;
            Vector3 sampleOrigin = new Vector3(
                bodyRoot.position.x,
                eyeCamera.transform.position.y + 2f,
                bodyRoot.position.z);
            int hitCount = Physics.RaycastNonAlloc(
                sampleOrigin,
                Vector3.down,
                hitBuffer,
                6f,
                raycastMask,
                QueryTriggerInteraction.Ignore);
            bool found = false;
            float lowestGround = float.PositiveInfinity;
            for (int i = 0; i < hitCount; i++)
            {
                RaycastHit candidate = hitBuffer[i];
                if (candidate.collider == null || IsSelf(candidate.collider) ||
                    candidate.normal.y < 0.5f)
                {
                    continue;
                }
                if (candidate.point.y < lowestGround)
                {
                    lowestGround = candidate.point.y;
                    found = true;
                }
            }
            if (found)
            {
                return lowestGround;
            }

            Collider rootCollider = bodyRoot.GetComponent<Collider>();
            return rootCollider != null
                ? rootCollider.bounds.min.y
                : bodyRoot.position.y;
        }

        bool IsSelf(Collider collider)
        {
            Transform bodyRoot = eyeCamera.transform.root;
            Transform candidate = collider.transform;
            return candidate == bodyRoot || candidate.IsChildOf(bodyRoot);
        }

        string ClassifySurface(RaycastHit hit, float groundY)
        {
            float horizontalNormal = Mathf.Sqrt(
                hit.normal.x * hit.normal.x + hit.normal.z * hit.normal.z);
            float slopeDegrees = Mathf.Atan2(
                horizontalNormal,
                Mathf.Max(1e-6f, Mathf.Abs(hit.normal.y))) * Mathf.Rad2Deg;
            if (hit.normal.y < 0f)
            {
                slopeDegrees = 180f - slopeDegrees;
            }

            float heightMeters = hit.point.y - groundY;
            if (hit.normal.y >= Mathf.Cos(maxSlopeDegrees * Mathf.Deg2Rad))
            {
                return heightMeters <= stepHeightMeters ? "floor" : "obstacle";
            }
            return heightMeters > 2f ? "overhead" : "obstacle";
        }
    }

    [Serializable]
    public struct NavigationRaySample
    {
        public Vector2 pixelUv;
        public Vector3 directionCamera;
        public bool hit;
        public float hitDistanceMeters;
        public Vector3 worldPosition;
        public Vector3 worldNormal;
        public string surface;
    }

    [Serializable]
    public sealed class NavigationSampleSet
    {
        public string source;
        public float groundY;
        public float maxSlopeDegrees;
        public float stepHeightMeters;
        public NavigationRaySample[] rays;
    }
}
