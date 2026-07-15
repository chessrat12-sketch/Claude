// Turns a WorldSnapshot into 3D actors: ground, resource props, shelters,
// predators, and agents. Holds no simulation state and makes no decisions —
// it renders whatever the snapshot says, exactly like the browser viewer and
// the Unity client do from the same JSON contract.
//
// Grid convention: 1 world grid cell = 100 Unreal units (cm), so a tile is
// about the size of a default 100x100x100 cube. Node/agent/hut actors are
// placed at (x*100, y*100, 0).

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "SnapshotModels.h"
#include "VillageManager.generated.h"

class UStaticMeshComponent;
class ADirectionalLight;
class AAgentActor;
class ANodeActor;
class AHutActor;
class AThreatActor;

UCLASS()
class AVillageManager : public AActor
{
	GENERATED_BODY()

public:
	AVillageManager();

	// Optional: assign a Directional Light in the level for day/night
	// intensity changes. If left empty, day/night is skipped gracefully.
	UPROPERTY(EditAnywhere, Category = "Emergent Civilization")
	ADirectionalLight* Sun;

	static constexpr float CellSize = 100.f;

	void ApplySnapshot(const FWorldSnapshot& Snapshot);

protected:
	virtual void Tick(float DeltaSeconds) override;

private:
	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* GroundMesh;
	bool bGroundBuilt = false;

	UPROPERTY() TMap<FString, ANodeActor*> Nodes;     // key: "x,y"
	UPROPERTY() TMap<FString, AHutActor*> Huts;       // key: "x,y"
	UPROPERTY() TMap<FString, AThreatActor*> Threats; // key: id

	struct FAgentEntry
	{
		AAgentActor* Actor = nullptr;
		FVector Target = FVector::ZeroVector;
	};
	TMap<FString, FAgentEntry> Agents; // key: agent id

	void BuildGround(int32 Width, int32 Height);
	void SyncNodes(const FWorldSnapshot& Snapshot);
	void SyncStructures(const FWorldSnapshot& Snapshot);
	void SyncThreats(const FWorldSnapshot& Snapshot);
	void SyncAgents(const FWorldSnapshot& Snapshot);
	void ApplyDayNight(bool bIsNight);

	static FVector GridToWorld(int32 GridX, int32 GridY);
};
