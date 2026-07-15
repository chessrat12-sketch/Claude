// A shelter prop: box base + cone roof.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "HutActor.generated.h"

class UStaticMeshComponent;

UCLASS()
class AHutActor : public AActor
{
	GENERATED_BODY()

public:
	AHutActor();

	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* Base;
	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* Roof;
};
