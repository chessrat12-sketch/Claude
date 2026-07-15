// A predator prop: small dark body + head, no label/health bar (keeps it
// visually distinct from agents at a glance).

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "ThreatActor.generated.h"

class UStaticMeshComponent;

UCLASS()
class AThreatActor : public AActor
{
	GENERATED_BODY()

public:
	AThreatActor();

	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* Body;
	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* Head;
};
