// Minimal orbit / zoom / pan camera so you can actually look around the 3D
// village. Left-drag orbits, scroll zooms, right-drag (or middle-drag) pans.
// Attached automatically by SceneBootstrap; can also be added by hand.

using UnityEngine;

namespace EmergentCivilization
{
    public class CameraOrbit : MonoBehaviour
    {
        public Vector3 target = Vector3.zero;
        public float distance = 18f;
        public float yaw = 45f;
        public float pitch = 50f;

        public float orbitSpeed = 4f;
        public float zoomSpeed = 8f;
        public float panSpeed = 0.5f;
        public float minDistance = 4f;
        public float maxDistance = 80f;

        private void Start() => Apply();

        private void LateUpdate()
        {
            // Orbit with left mouse.
            if (Input.GetMouseButton(0))
            {
                yaw += Input.GetAxis("Mouse X") * orbitSpeed;
                pitch = Mathf.Clamp(pitch - Input.GetAxis("Mouse Y") * orbitSpeed, 10f, 85f);
            }
            // Pan with right / middle mouse.
            if (Input.GetMouseButton(1) || Input.GetMouseButton(2))
            {
                Vector3 move = (-transform.right * Input.GetAxis("Mouse X")
                                - transform.up * Input.GetAxis("Mouse Y")) * panSpeed;
                target += move;
            }
            // Zoom with the scroll wheel.
            distance = Mathf.Clamp(distance - Input.GetAxis("Mouse ScrollWheel") * zoomSpeed * 10f,
                                   minDistance, maxDistance);
            Apply();
        }

        private void Apply()
        {
            Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);
            transform.position = target + rot * new Vector3(0f, 0f, -distance);
            transform.LookAt(target);
        }
    }
}
