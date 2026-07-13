// Polls the Python viz server for world snapshots and hands them to a renderer.
//
// Attach to an empty GameObject, set `serverUrl`, and point `renderer` at a
// VillageRenderer. Nothing here decides agent behaviour — it is a pure client
// of the simulation running in Python (the single source of truth).

using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace EmergentCivilization
{
    public class SimClient : MonoBehaviour
    {
        [Tooltip("Base URL of the Python viz server (server/viz_server.py).")]
        public string serverUrl = "http://localhost:8000";

        [Tooltip("Seconds between snapshot polls. Match the server's tick rate.")]
        public float pollInterval = 0.3f;

        public VillageRenderer villageRenderer;

        private int _lastTick = -1;

        private void Start()
        {
            StartCoroutine(PollLoop());
        }

        private IEnumerator PollLoop()
        {
            while (true)
            {
                using (UnityWebRequest req = UnityWebRequest.Get(serverUrl + "/state"))
                {
                    yield return req.SendWebRequest();

                    if (req.result == UnityWebRequest.Result.Success)
                    {
                        WorldSnapshot snap = JsonUtility.FromJson<WorldSnapshot>(req.downloadHandler.text);
                        if (snap != null && snap.tick != _lastTick)
                        {
                            _lastTick = snap.tick;
                            if (villageRenderer != null)
                                villageRenderer.Apply(snap);
                        }
                    }
                    else
                    {
                        Debug.LogWarning($"[SimClient] {req.error} — is viz_server.py running?");
                    }
                }

                yield return new WaitForSeconds(pollInterval);
            }
        }
    }
}
