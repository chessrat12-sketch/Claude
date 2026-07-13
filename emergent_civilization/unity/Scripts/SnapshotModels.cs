// Serializable mirrors of the Python render snapshot (sim/snapshot.py).
//
// Field names match the JSON keys exactly so UnityEngine.JsonUtility can parse
// a snapshot with no third-party JSON library. The schema deliberately uses
// arrays instead of maps (e.g. inventory is a list of {resource, amount}),
// because JsonUtility cannot deserialise dictionaries.

using System;

namespace EmergentCivilization
{
    [Serializable]
    public class InventoryEntry
    {
        public string resource;
        public int amount;
    }

    [Serializable]
    public class NodeView
    {
        public int x;
        public int y;
        public string resource;   // "food" | "wood" | "stone"
        public int amount;
        public int capacity;
    }

    [Serializable]
    public class AgentView
    {
        public string id;
        public string name;
        public int x;
        public int y;
        public int hunger;
        public int energy;
        public int health;
        public bool alive;
        public int wealth;
        public string lastAction;
        public InventoryEntry[] inventory;
    }

    [Serializable]
    public class EventView
    {
        public string type;       // "trade" | "speak"
        public string a;
        public string b;
        public string text;
    }

    [Serializable]
    public class StatsView
    {
        public int tick;
        public int alive;
        public int population;
        public int deaths;
        public int trades;
        public int messages;
        public float gini;
        // priceWoodForFood may be null in JSON; JsonUtility leaves it 0 then.
        public float priceWoodForFood;
    }

    [Serializable]
    public class WorldSnapshot
    {
        public int tick;
        public int width;
        public int height;
        public NodeView[] nodes;
        public AgentView[] agents;
        public EventView[] events;
        public StatsView stats;
    }
}
