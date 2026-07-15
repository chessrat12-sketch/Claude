// Serializable mirrors of the Python render snapshot (sim/snapshot.py).
//
// Field names match the JSON keys exactly so FJsonObjectConverter can
// deserialise a snapshot straight into these structs with no manual parsing.

#pragma once

#include "CoreMinimal.h"
#include "SnapshotModels.generated.h"

USTRUCT(BlueprintType)
struct FInventoryEntry
{
	GENERATED_BODY()

	UPROPERTY() FString resource;
	UPROPERTY() int32 amount = 0;
};

USTRUCT(BlueprintType)
struct FNodeView
{
	GENERATED_BODY()

	UPROPERTY() int32 x = 0;
	UPROPERTY() int32 y = 0;
	UPROPERTY() FString resource; // "food" | "wood" | "stone"
	UPROPERTY() int32 amount = 0;
	UPROPERTY() int32 capacity = 0;
};

USTRUCT(BlueprintType)
struct FStructureView
{
	GENERATED_BODY()

	UPROPERTY() int32 x = 0;
	UPROPERTY() int32 y = 0;
	UPROPERTY() FString owner;
	UPROPERTY() int32 durability = 0;
};

USTRUCT(BlueprintType)
struct FThreatView
{
	GENERATED_BODY()

	UPROPERTY() FString id;
	UPROPERTY() int32 x = 0;
	UPROPERTY() int32 y = 0;
};

USTRUCT(BlueprintType)
struct FAgentView
{
	GENERATED_BODY()

	UPROPERTY() FString id;
	UPROPERTY() FString name;
	UPROPERTY() int32 x = 0;
	UPROPERTY() int32 y = 0;
	UPROPERTY() int32 hunger = 0;
	UPROPERTY() int32 energy = 0;
	UPROPERTY() int32 health = 0;
	UPROPERTY() bool alive = true;
	UPROPERTY() int32 wealth = 0;
	UPROPERTY() bool sheltered = false;
	UPROPERTY() int32 tools = 0;
	UPROPERTY() FString lastAction;
	UPROPERTY() TArray<FInventoryEntry> inventory;
};

USTRUCT(BlueprintType)
struct FEventView
{
	GENERATED_BODY()

	UPROPERTY() FString type; // "trade" | "speak" | "gift" | "build" | "attack" | "thought"
	UPROPERTY() FString a;
	UPROPERTY() FString b;
	UPROPERTY() FString text;
};

USTRUCT(BlueprintType)
struct FStatsView
{
	GENERATED_BODY()

	UPROPERTY() int32 tick = 0;
	UPROPERTY() int32 alive = 0;
	UPROPERTY() int32 population = 0;
	UPROPERTY() int32 deaths = 0;
	UPROPERTY() int32 trades = 0;
	UPROPERTY() int32 gifts = 0;
	UPROPERTY() int32 shelters = 0;
	UPROPERTY() int32 tools = 0;
	UPROPERTY() int32 attacks = 0;
	UPROPERTY() int32 alliances = 0;
	UPROPERTY() float meanTrust = 0.f;
	UPROPERTY() float gini = 0.f;
	UPROPERTY() float priceWoodForFood = 0.f;
};

USTRUCT(BlueprintType)
struct FWorldSnapshot
{
	GENERATED_BODY()

	UPROPERTY() int32 tick = 0;
	UPROPERTY() int32 width = 0;
	UPROPERTY() int32 height = 0;
	UPROPERTY() int32 timeOfDay = 0;
	UPROPERTY() int32 dayLength = 0;
	UPROPERTY() bool isNight = false;
	UPROPERTY() TArray<FNodeView> nodes;
	UPROPERTY() TArray<FStructureView> structures;
	UPROPERTY() TArray<FThreatView> threats;
	UPROPERTY() TArray<FAgentView> agents;
	UPROPERTY() TArray<FEventView> events;
	UPROPERTY() FStatsView stats;
};
