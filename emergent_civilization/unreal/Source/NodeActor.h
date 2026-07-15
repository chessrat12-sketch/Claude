// A resource node prop: a tree (trunk+leaf) for food/wood, or a rock for
// stone. Both mesh sets exist on the actor; SetResourceType toggles which is
// visible, so no dynamic material work is needed to tell resources apart.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "NodeActor.generated.h"

class UStaticMeshComponent;

UCLASS()
class ANodeActor : public AActor
{
	GENERATED_BODY()

public:
	ANodeActor();

	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* Trunk;
	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* Leaf;
	UPROPERTY(VisibleAnywhere) UStaticMeshComponent* Rock;

	void SetResourceType(const FString& Resource);
	// fraction in [0,1] of amount/capacity — scales the visible prop.
	void SetFill(float Fraction);
};
