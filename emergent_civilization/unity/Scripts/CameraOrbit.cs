// Minimal orbit / zoom / pan camera so you can actually look around the 3D
// village. Left-drag orbits, scroll zooms, right-drag (or middle-drag) pans.
// Attached automatically by SceneBootstrap; can also be added by hand.
//
// Supports both of Unity's input backends (Project Settings > Player > Active
// Input Handling): the legacy UnityEngine.Input class, and the newer Input
// System package. Unity defines ENABLE_INPUT_SYSTEM automatically when that
// package is active, so this picks the right one at compile time — no manual
// setup needed either way. (If a project is set to "Input System Package
// (New)" only, the legacy Input class throws InvalidOperationException; this
// avoids that entirely rather than requiring a Project Settings change.)

using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

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
            bool orbitHeld, panHeld;
            float dx, dy, scroll;

#if ENABLE_INPUT_SYSTEM
            var mouse = Mouse.current;
            if (mouse == null) return;
            orbitHeld = mouse.leftButton.isPressed;
            panHeld = mouse.rightButton.isPressed || mouse.middleButton.isPressed;
            Vector2 delta = mouse.delta.ReadValue();
            // The new Input System reports raw pixel deltas (much larger than
            // legacy's normalised GetAxis values), and scroll in ~120-per-notch
            // steps — scale both down so sensitivity feels the same as before.
            dx = delta.x * 0.02f;
            dy = delta.y * 0.02f;
            scroll = mouse.scroll.ReadValue().y * 0.01f;
#else
            orbitHeld = Input.GetMouseButton(0);
            panHeld = Input.GetMouseButton(1) || Input.GetMouseButton(2);
            dx = Input.GetAxis("Mouse X");
            dy = Input.GetAxis("Mouse Y");
            scroll = Input.GetAxis("Mouse ScrollWheel");
#endif

            // Orbit with left mouse.
            if (orbitHeld)
            {
                yaw += dx * orbitSpeed;
                pitch = Mathf.Clamp(pitch - dy * orbitSpeed, 10f, 85f);
            }
            // Pan with right / middle mouse.
            if (panHeld)
            {
                Vector3 move = (-transform.right * dx - transform.up * dy) * panSpeed;
                target += move;
            }
            // Zoom with the scroll wheel.
            distance = Mathf.Clamp(distance - scroll * zoomSpeed * 10f, minDistance, maxDistance);
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
