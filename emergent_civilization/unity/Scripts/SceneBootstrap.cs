// One-click scene setup. Add THIS single component to one empty GameObject in a
// fresh 3D scene, press Play, and it constructs and wires everything else:
// a camera (with orbit controls), a sun for day/night, the ground+renderer, and
// the SimClient that streams from the Python viz server. No manual wiring, no
// prefabs required.
//
//   1. Run the server:  python -m server.viz_server
//   2. Unity: empty GameObject -> Add Component -> "Scene Bootstrap" -> Play.

using UnityEngine;

namespace EmergentCivilization
{
    public class SceneBootstrap : MonoBehaviour
    {
        [Tooltip("Base URL of the Python viz server (server/viz_server.py).")]
        public string serverUrl = "http://localhost:8000";

        [Tooltip("Grid size of the world you launched the server with (--size).")]
        public int worldSize = 14;

        [Tooltip("Seconds between snapshot polls; match the server's --tick-ms.")]
        public float pollInterval = 0.3f;

        private void Awake()
        {
            float c = (worldSize - 1) / 2f;                 // world centre
            Vector3 centre = new Vector3(c, 0f, c);

            // --- Camera (with orbit controls) --------------------------------
            Camera cam = Camera.main;
            if (cam == null)
            {
                GameObject camGo = new GameObject("Main Camera");
                camGo.tag = "MainCamera";
                cam = camGo.AddComponent<Camera>();
                camGo.AddComponent<AudioListener>();
            }
            cam.clearFlags = CameraClearFlags.SolidColor;
            var orbit = cam.gameObject.GetComponent<CameraOrbit>() ?? cam.gameObject.AddComponent<CameraOrbit>();
            orbit.target = centre;
            orbit.distance = worldSize * 1.3f;

            // --- Sun (for day/night) -----------------------------------------
            GameObject sunGo = new GameObject("Sun");
            Light sun = sunGo.AddComponent<Light>();
            sun.type = LightType.Directional;
            sunGo.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            // --- Village renderer --------------------------------------------
            GameObject village = new GameObject("Village");
            var renderer = village.AddComponent<VillageRenderer>();
            renderer.sun = sun;
            renderer.targetCamera = cam;

            // --- Sim client (streams from the server) ------------------------
            GameObject clientGo = new GameObject("SimClient");
            var client = clientGo.AddComponent<SimClient>();
            client.serverUrl = serverUrl;
            client.pollInterval = pollInterval;
            client.villageRenderer = renderer;

            Debug.Log($"[EmergentCivilization] scene ready — streaming from {serverUrl}. "
                      + "Drag to orbit, scroll to zoom.");
        }
    }
}
