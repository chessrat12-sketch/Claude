// One agent: a body + head (primitive stand-ins until real character meshes
// are added), a floating name label, and a floating health bar. Everything
// here is transform-only (position/scale) — no material parameter tricks —
// so it's robust across any project's default material setup.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AgentActor.generated.h"

class UStaticMeshComponent;
class UTextRenderComponent;

UCLASS()
class AAgentActor : public AActor
{
	GENERATED_BODY()

public:
	AAgentActor();

	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* Body;
	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* Head;
	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* HealthBarBack;
	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* HealthBarFill;
	UPROPERTY(VisibleAnywhere) UTextRenderComponent* NameLabel;

	// health in [0,100]; sets the bar's fill scale and colour.
	void SetHealth(int32 Health);
	void SetName(const FString& InName);

	// Called every frame by the manager so labels/bars always face the camera.
	void FaceCamera(const FRotator& CameraRotation);
};
