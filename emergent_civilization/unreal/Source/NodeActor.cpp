#include "NodeActor.h"
#include "Components/StaticMeshComponent.h"
#include "UObject/ConstructorHelpers.h"

ANodeActor::ANodeActor()
{
	PrimaryActorTick.bCanEverTick = false;

	static ConstructorHelpers::FObjectFinder<UStaticMesh> CylinderAsset(
		TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> ConeAsset(
		TEXT("/Engine/BasicShapes/Cone.Cone"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereAsset(
		TEXT("/Engine/BasicShapes/Sphere.Sphere"));

	Trunk = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Trunk"));
	RootComponent = Trunk;
	if (CylinderAsset.Succeeded()) Trunk->SetStaticMesh(CylinderAsset.Object);
	Trunk->SetRelativeScale3D(FVector(0.15f, 0.15f, 0.5f));
	Trunk->SetRelativeLocation(FVector(0.f, 0.f, 25.f));
	Trunk->SetMobility(EComponentMobility::Movable);
	Trunk->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	Leaf = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Leaf"));
	Leaf->SetupAttachment(Trunk);
	if (ConeAsset.Succeeded()) Leaf->SetStaticMesh(ConeAsset.Object);
	Leaf->SetRelativeScale3D(FVector(0.6f, 0.6f, 0.9f));
	Leaf->SetRelativeLocation(FVector(0.f, 0.f, 75.f));
	Leaf->SetMobility(EComponentMobility::Movable);
	Leaf->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	Rock = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Rock"));
	Rock->SetupAttachment(Trunk);
	if (SphereAsset.Succeeded()) Rock->SetStaticMesh(SphereAsset.Object);
	Rock->SetRelativeScale3D(FVector(0.4f, 0.4f, 0.3f));
	Rock->SetRelativeLocation(FVector(0.f, 0.f, 15.f));
	Rock->SetMobility(EComponentMobility::Movable);
	Rock->SetCollisionEnabled(ECollisionEnabled::NoCollision);
}

void ANodeActor::SetResourceType(const FString& Resource)
{
	const bool bIsStone = Resource == TEXT("stone");
	Trunk->SetVisibility(!bIsStone);
	Leaf->SetVisibility(!bIsStone);
	Rock->SetVisibility(bIsStone);
}

void ANodeActor::SetFill(float Fraction)
{
	const float Scale = FMath::Clamp(0.5f + 0.5f * Fraction, 0.2f, 1.f);
	SetActorScale3D(FVector(Scale));
}
