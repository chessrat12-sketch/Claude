#include "VillageManager.h"
#include "AgentActor.h"
#include "NodeActor.h"
#include "HutActor.h"
#include "ThreatActor.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/DirectionalLight.h"
#include "Components/LightComponent.h"
#include "Kismet/GameplayStatics.h"
#include "UObject/ConstructorHelpers.h"

AVillageManager::AVillageManager()
{
	PrimaryActorTick.bCanEverTick = true;

	static ConstructorHelpers::FObjectFinder<UStaticMesh> PlaneAsset(
		TEXT("/Engine/BasicShapes/Plane.Plane"));

	GroundMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("GroundMesh"));
	RootComponent = GroundMesh;
	if (PlaneAsset.Succeeded()) GroundMesh->SetStaticMesh(PlaneAsset.Object);
	GroundMesh->SetMobility(EComponentMobility::Movable);
	GroundMesh->SetVisibility(false); // shown once sized in BuildGround
}

FVector AVillageManager::GridToWorld(int32 GridX, int32 GridY)
{
	return FVector(GridX * CellSize, GridY * CellSize, 0.f);
}

void AVillageManager::BuildGround(int32 Width, int32 Height)
{
	// Engine's basic Plane is ~100x100 units at scale 1, matching CellSize —
	// so scaling by (Width, Height, 1) covers the grid, centred under it.
	GroundMesh->SetVisibility(true);
	GroundMesh->SetWorldScale3D(FVector(Width, Height, 1.f));
	GroundMesh->SetWorldLocation(FVector((Width - 1) * CellSize * 0.5f,
	                                      (Height - 1) * CellSize * 0.5f, 0.f));
	bGroundBuilt = true;
}

void AVillageManager::ApplySnapshot(const FWorldSnapshot& Snapshot)
{
	if (!bGroundBuilt) BuildGround(Snapshot.width, Snapshot.height);
	ApplyDayNight(Snapshot.isNight);
	SyncNodes(Snapshot);
	SyncStructures(Snapshot);
	SyncThreats(Snapshot);
	SyncAgents(Snapshot);
}

void AVillageManager::ApplyDayNight(bool bIsNight)
{
	if (!Sun) return;
	ULightComponent* Light = Sun->GetLightComponent();
	if (!Light) return;
	Light->SetIntensity(bIsNight ? 0.3f : 3.0f);
	Light->SetLightColor(bIsNight ? FLinearColor(0.55f, 0.6f, 0.9f) : FLinearColor::White);
}

void AVillageManager::SyncNodes(const FWorldSnapshot& Snapshot)
{
	TSet<FString> Seen;
	for (const FNodeView& Node : Snapshot.nodes)
	{
		const FString Key = FString::Printf(TEXT("%d,%d"), Node.x, Node.y);
		Seen.Add(Key);
		ANodeActor** Existing = Nodes.Find(Key);
		ANodeActor* NodeActorPtr;
		if (Existing)
		{
			NodeActorPtr = *Existing;
		}
		else
		{
			NodeActorPtr = GetWorld()->SpawnActor<ANodeActor>(GridToWorld(Node.x, Node.y), FRotator::ZeroRotator);
			Nodes.Add(Key, NodeActorPtr);
		}
		if (NodeActorPtr)
		{
			NodeActorPtr->SetResourceType(Node.resource);
			const float Fraction = Node.capacity > 0 ? (float)Node.amount / Node.capacity : 1.f;
			NodeActorPtr->SetFill(Fraction);
		}
	}
	for (auto It = Nodes.CreateIterator(); It; ++It)
	{
		if (!Seen.Contains(It->Key))
		{
			if (It->Value) It->Value->Destroy();
			It.RemoveCurrent();
		}
	}
}

void AVillageManager::SyncStructures(const FWorldSnapshot& Snapshot)
{
	TSet<FString> Seen;
	for (const FStructureView& Structure : Snapshot.structures)
	{
		const FString Key = FString::Printf(TEXT("%d,%d"), Structure.x, Structure.y);
		Seen.Add(Key);
		if (!Huts.Contains(Key))
		{
			AHutActor* Hut = GetWorld()->SpawnActor<AHutActor>(
				GridToWorld(Structure.x, Structure.y), FRotator::ZeroRotator);
			Huts.Add(Key, Hut);
		}
	}
	for (auto It = Huts.CreateIterator(); It; ++It)
	{
		if (!Seen.Contains(It->Key))
		{
			if (It->Value) It->Value->Destroy();
			It.RemoveCurrent();
		}
	}
}

void AVillageManager::SyncThreats(const FWorldSnapshot& Snapshot)
{
	TSet<FString> Seen;
	for (const FThreatView& Threat : Snapshot.threats)
	{
		Seen.Add(Threat.id);
		AThreatActor** Existing = Threats.Find(Threat.id);
		AThreatActor* ThreatActorPtr;
		if (Existing)
		{
			ThreatActorPtr = *Existing;
		}
		else
		{
			ThreatActorPtr = GetWorld()->SpawnActor<AThreatActor>(
				GridToWorld(Threat.x, Threat.y), FRotator::ZeroRotator);
			Threats.Add(Threat.id, ThreatActorPtr);
		}
		if (ThreatActorPtr)
		{
			ThreatActorPtr->SetActorLocation(GridToWorld(Threat.x, Threat.y));
		}
	}
	for (auto It = Threats.CreateIterator(); It; ++It)
	{
		if (!Seen.Contains(It->Key))
		{
			if (It->Value) It->Value->Destroy();
			It.RemoveCurrent();
		}
	}
}

void AVillageManager::SyncAgents(const FWorldSnapshot& Snapshot)
{
	for (const FAgentView& Agent : Snapshot.agents)
	{
		FAgentEntry* Existing = Agents.Find(Agent.id);
		if (!Existing)
		{
			AAgentActor* NewActor = GetWorld()->SpawnActor<AAgentActor>(
				GridToWorld(Agent.x, Agent.y), FRotator::ZeroRotator);
			NewActor->SetName(Agent.name);
			FAgentEntry Entry;
			Entry.Actor = NewActor;
			Entry.Target = GridToWorld(Agent.x, Agent.y);
			Agents.Add(Agent.id, Entry);
			Existing = Agents.Find(Agent.id);
		}
		Existing->Target = GridToWorld(Agent.x, Agent.y);
		if (Existing->Actor)
		{
			Existing->Actor->SetActorHiddenInGame(!Agent.alive);
			Existing->Actor->SetHealth(Agent.health);
		}
	}
	// Agents never leave the roster (they can die but stay visible/hidden),
	// so no removal pass here — unlike nodes/huts/threats which can vanish.
}

void AVillageManager::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);

	FRotator CameraRotation = FRotator::ZeroRotator;
	if (APlayerController* PC = UGameplayStatics::GetPlayerController(this, 0))
	{
		if (PC->PlayerCameraManager)
		{
			CameraRotation = PC->PlayerCameraManager->GetCameraRotation();
		}
	}

	for (auto& Pair : Agents)
	{
		FAgentEntry& Entry = Pair.Value;
		if (!Entry.Actor) continue;
		const FVector Current = Entry.Actor->GetActorLocation();
		const FVector NewLocation = FMath::VInterpTo(Current, Entry.Target, DeltaSeconds, 4.f);
		Entry.Actor->SetActorLocation(NewLocation);
		Entry.Actor->FaceCamera(CameraRotation);
	}
}
