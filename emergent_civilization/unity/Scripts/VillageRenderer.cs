// Turns a WorldSnapshot into 3D GameObjects: a ground plane, resource props
// (trees / rocks), and agent avatars that glide to their new grid cells.
//
// It renders whatever the snapshot says and no more — it holds no simulation
// state and makes no decisions. Prefabs are optional: if you leave them unset
// the renderer builds primitives (cubes/cylinders/spheres) so the scene works
// with zero art. One world unit = one grid cell.

using System.Collections.Generic;
using UnityEngine;

namespace EmergentCivilization
{
    public class VillageRenderer : MonoBehaviour
    {
        [Header("Optional prefabs (primitives used if empty)")]
        public GameObject treePrefab;    // for food/wood nodes
        public GameObject rockPrefab;    // for stone nodes
        public GameObject agentPrefab;   // for agents

        [Header("Layout")]
        public float glideSpeed = 6f;    // how fast avatars slide between cells

        private readonly Dictionary<string, Transform> _agents = new();
        private readonly Dictionary<string, Vector3> _targets = new();
        private readonly Dictionary<string, GameObject> _nodes = new();
        private bool _groundBuilt;

        public void Apply(WorldSnapshot snap)
        {
            if (!_groundBuilt) BuildGround(snap.width, snap.height);
            SyncNodes(snap);
            SyncAgents(snap);
        }

        private void Update()
        {
            // Smoothly move avatars toward their latest target cell.
            foreach (var kv in _agents)
            {
                if (_targets.TryGetValue(kv.Key, out Vector3 target))
                    kv.Value.position = Vector3.MoveTowards(
                        kv.Value.position, target, glideSpeed * Time.deltaTime);
            }
        }

        private void BuildGround(int w, int h)
        {
            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(transform, false);
            // Unity's plane is 10x10 units, so scale by size/10 and centre it.
            ground.transform.localScale = new Vector3(w / 10f, 1f, h / 10f);
            ground.transform.position = new Vector3((w - 1) / 2f, 0f, (h - 1) / 2f);
            var mr = ground.GetComponent<Renderer>();
            mr.material.color = new Color(0.23f, 0.45f, 0.28f);
            _groundBuilt = true;
        }

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
                // Scale a node prop by how full it is.
                float fill = n.capacity > 0 ? (float)n.amount / n.capacity : 1f;
                go.transform.localScale = Vector3.one * (0.5f + 0.5f * fill);
            }
            // Remove depleted nodes.
            var stale = new List<string>();
            foreach (var kv in _nodes)
                if (!seen.Contains(kv.Key)) { Destroy(kv.Value); stale.Add(kv.Key); }
            foreach (var k in stale) _nodes.Remove(k);
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
                    t = go.transform;
                    t.position = new Vector3(a.x, 0.5f, a.y);
                    _agents[a.id] = t;
                }
                _targets[a.id] = new Vector3(a.x, 0.5f, a.y);
                t.gameObject.SetActive(a.alive);
            }
        }

        private static Color ColorFor(string id)
        {
            int h = 0;
            foreach (char c in id) h = (h * 31 + c) % 360;
            return Color.HSVToRGB(h / 360f, 0.55f, 0.85f);
        }
    }
}
