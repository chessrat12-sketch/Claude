// Turns a WorldSnapshot into a 3D village: ground, resource props (trees/rocks),
// shelters (huts), roaming predators, and agent avatars with floating health
// bars. Day/night is reflected in the lighting, and social events (speak, gift,
// trade, build, predator bites) pop up as short-lived floating labels.
//
// It renders whatever the snapshot says and no more — it holds no simulation
// state and makes no decisions. Prefabs are optional: if you leave them unset
// the renderer builds primitives so the scene works with zero art. One world
// unit = one grid cell.

using System.Collections.Generic;
using UnityEngine;

namespace EmergentCivilization
{
    public class VillageRenderer : MonoBehaviour
    {
        [Header("Optional prefabs (primitives used if empty)")]
        public GameObject treePrefab;      // for food/wood nodes
        public GameObject rockPrefab;      // for stone nodes
        public GameObject agentPrefab;     // for agents
        public GameObject shelterPrefab;   // for shelters
        public GameObject predatorPrefab;  // for predators

        [Header("Layout & mood")]
        public float glideSpeed = 6f;
        public Light sun;                  // optional directional light for day/night
        public Camera targetCamera;        // defaults to Camera.main

        private readonly Dictionary<string, Transform> _agents = new();
        private readonly Dictionary<string, Vector3> _targets = new();
        private readonly Dictionary<string, GameObject> _nodes = new();
        private readonly Dictionary<string, GameObject> _huts = new();
        private readonly Dictionary<string, GameObject> _threats = new();
        private bool _groundBuilt;

        public void Apply(WorldSnapshot snap)
        {
            if (!_groundBuilt) BuildGround(snap.width, snap.height);
            ApplyDayNight(snap.isNight);
            SyncNodes(snap);
            SyncStructures(snap);
            SyncThreats(snap);
            SyncAgents(snap);
            ShowEvents(snap);
        }

        private void Update()
        {
            foreach (var kv in _agents)
                if (_targets.TryGetValue(kv.Key, out Vector3 target))
                    kv.Value.position = Vector3.MoveTowards(
                        kv.Value.position, target, glideSpeed * Time.deltaTime);
        }

        // -- environment ------------------------------------------------------
        private void BuildGround(int w, int h)
        {
            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(transform, false);
            ground.transform.localScale = new Vector3(w / 10f, 1f, h / 10f);
            ground.transform.position = new Vector3((w - 1) / 2f, 0f, (h - 1) / 2f);
            ground.GetComponent<Renderer>().material.color = new Color(0.23f, 0.45f, 0.28f);
            _groundBuilt = true;
        }

        private void ApplyDayNight(bool night)
        {
            Camera cam = targetCamera != null ? targetCamera : Camera.main;
            if (cam != null)
                cam.backgroundColor = night ? new Color(0.05f, 0.07f, 0.15f)
                                            : new Color(0.53f, 0.71f, 0.92f);
            RenderSettings.ambientLight = night ? new Color(0.2f, 0.24f, 0.36f)
                                                : new Color(0.8f, 0.8f, 0.75f);
            if (sun != null) sun.intensity = night ? 0.25f : 1f;
        }

        // -- props ------------------------------------------------------------
        private void SyncNodes(WorldSnapshot snap)
        {
            var seen = new HashSet<string>();
            foreach (NodeView n in snap.nodes)
            {
                string key = $"{n.x},{n.y}";
                seen.Add(key);
                if (!_nodes.TryGetValue(key, out GameObject go))
                {
                    go = SpawnNode(n);
                    _nodes[key] = go;
                }
                float fill = n.capacity > 0 ? (float)n.amount / n.capacity : 1f;
                go.transform.localScale = Vector3.one * (0.5f + 0.5f * fill);
            }
            Prune(_nodes, seen);
        }

        private GameObject SpawnNode(NodeView n)
        {
            bool isStone = n.resource == "stone";
            GameObject prefab = isStone ? rockPrefab : treePrefab;
            GameObject go = prefab != null
                ? Instantiate(prefab)
                : GameObject.CreatePrimitive(isStone ? PrimitiveType.Sphere : PrimitiveType.Cylinder);
            go.name = $"{n.resource}_{n.x}_{n.y}";
            go.transform.SetParent(transform, false);
            go.transform.position = new Vector3(n.x, 0.5f, n.y);
            if (prefab == null)
            {
                Color c = n.resource == "food" ? new Color(0.36f, 0.75f, 0.42f)
                        : n.resource == "wood" ? new Color(0.5f, 0.33f, 0.19f)
                        : new Color(0.6f, 0.64f, 0.7f);
                go.GetComponent<Renderer>().material.color = c;
            }
            return go;
        }

        private void SyncStructures(WorldSnapshot snap)
        {
            var seen = new HashSet<string>();
            foreach (StructureView s in snap.structures ?? new StructureView[0])
            {
                string key = $"{s.x},{s.y}";
                seen.Add(key);
                if (_huts.ContainsKey(key)) continue;
                GameObject go = shelterPrefab != null
                    ? Instantiate(shelterPrefab)
                    : GameObject.CreatePrimitive(PrimitiveType.Cube);
                go.name = $"shelter_{key}";
                go.transform.SetParent(transform, false);
                go.transform.position = new Vector3(s.x, 0.4f, s.y);
                go.transform.localScale = new Vector3(0.8f, 0.8f, 0.8f);
                if (shelterPrefab == null)
                    go.GetComponent<Renderer>().material.color = new Color(0.7f, 0.28f, 0.18f);
                _huts[key] = go;
            }
            Prune(_huts, seen);
        }

        private void SyncThreats(WorldSnapshot snap)
        {
            var seen = new HashSet<string>();
            foreach (ThreatView t in snap.threats ?? new ThreatView[0])
            {
                seen.Add(t.id);
                if (!_threats.TryGetValue(t.id, out GameObject go))
                {
                    go = predatorPrefab != null
                        ? Instantiate(predatorPrefab)
                        : GameObject.CreatePrimitive(PrimitiveType.Capsule);
                    go.name = $"predator_{t.id}";
                    go.transform.SetParent(transform, false);
                    go.transform.localScale = new Vector3(0.5f, 0.35f, 0.5f);
                    if (predatorPrefab == null)
                        go.GetComponent<Renderer>().material.color = new Color(0.22f, 0.24f, 0.29f);
                    _threats[t.id] = go;
                }
                go.transform.position = new Vector3(t.x, 0.4f, t.y);
            }
            Prune(_threats, seen);
        }

        // -- agents -----------------------------------------------------------
        private void SyncAgents(WorldSnapshot snap)
        {
            foreach (AgentView a in snap.agents)
            {
                if (!_agents.TryGetValue(a.id, out Transform t))
                {
                    GameObject go = agentPrefab != null
                        ? Instantiate(agentPrefab)
                        : GameObject.CreatePrimitive(PrimitiveType.Capsule);
                    go.name = a.name;
                    go.transform.SetParent(transform, false);
                    go.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
                    if (agentPrefab == null)
                        go.GetComponent<Renderer>().material.color = ColorFor(a.id);
                    AddHealthBar(go.transform);
                    t = go.transform;
                    t.position = new Vector3(a.x, 0.5f, a.y);
                    _agents[a.id] = t;
                }
                _targets[a.id] = new Vector3(a.x, 0.5f, a.y);
                t.gameObject.SetActive(a.alive);
                UpdateHealthBar(t, a.health);
            }
        }

        private static void AddHealthBar(Transform parent)
        {
            GameObject bar = GameObject.CreatePrimitive(PrimitiveType.Cube);
            bar.name = "HealthBar";
            Destroy(bar.GetComponent<Collider>());
            bar.transform.SetParent(parent, false);
            bar.transform.localPosition = new Vector3(0f, 2.4f, 0f);
            bar.transform.localScale = new Vector3(1.6f, 0.2f, 0.2f);
        }

        private static void UpdateHealthBar(Transform agent, int health)
        {
            Transform bar = agent.Find("HealthBar");
            if (bar == null) return;
            float t = Mathf.Clamp01(health / 100f);
            bar.localScale = new Vector3(1.6f * Mathf.Max(0.02f, t), 0.2f, 0.2f);
            var r = bar.GetComponent<Renderer>();
            if (r != null)
                r.material.color = t > 0.5f ? Color.green : t > 0.25f ? Color.yellow : Color.red;
        }

        // -- events -----------------------------------------------------------
        private void ShowEvents(WorldSnapshot snap)
        {
            foreach (EventView e in snap.events ?? new EventView[0])
            {
                Vector3 at;
                if (_agents.TryGetValue(e.b, out Transform tb)) at = tb.position;
                else if (_agents.TryGetValue(e.a, out Transform ta)) at = ta.position;
                else continue;
                FloatingLabel(at + Vector3.up * 2.8f, Icon(e.type) + (e.text ?? e.type));
            }
        }

        private static string Icon(string type) => type switch
        {
            "trade" => "\U0001F91D ", "gift" => "\U0001F381 ", "build" => "\U0001F3E0 ",
            "attack" => "\U0001F43A ", "speak" => "\U0001F4AC ", "thought" => "\U0001F4AD ",
            _ => "" };

        private void FloatingLabel(Vector3 pos, string text)
        {
            GameObject go = new GameObject("EventLabel");
            go.transform.SetParent(transform, false);
            go.transform.position = pos;
            var tm = go.AddComponent<TextMesh>();
            tm.text = text; tm.characterSize = 0.12f; tm.fontSize = 48;
            tm.anchor = TextAnchor.LowerCenter; tm.color = Color.white;
            if (Camera.main != null) go.transform.rotation = Camera.main.transform.rotation;
            Destroy(go, 2f);
        }

        // -- helpers ----------------------------------------------------------
        private static void Prune(Dictionary<string, GameObject> pool, HashSet<string> seen)
        {
            var stale = new List<string>();
            foreach (var kv in pool)
                if (!seen.Contains(kv.Key)) { Destroy(kv.Value); stale.Add(kv.Key); }
            foreach (var k in stale) pool.Remove(k);
        }

        private static Color ColorFor(string id)
        {
            int h = 0;
            foreach (char c in id) h = (h * 31 + c) % 360;
            return Color.HSVToRGB(h / 360f, 0.55f, 0.85f);
        }
    }
}
