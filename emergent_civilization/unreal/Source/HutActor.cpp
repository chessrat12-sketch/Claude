#include "HutActor.h"
#include "Components/StaticMeshComponent.h"
#include "UObject/ConstructorHelpers.h"

AHutActor::AHutActor()
{
	PrimaryActorTick.bCanEverTick = false;

	static ConstructorHelpers::FObjectFinder<UStaticMesh> CubeAsset(
		TEXT("/Engine/BasicShapes/Cube.Cube"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> ConeAsset(
		TEXT("/Engine/BasicShapes/Cone.Cone"));

	Base = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Base"));
	RootComponent = Base;
	if (CubeAsset.Succeeded()) Base->SetStaticMesh(CubeAsset.Object);
	Base->SetRelativeScale3D(FVector(0.8f, 0.8f, 0.6f));
	Base->SetRelativeLocation(FVector(0.f, 0.f, 30.f));
	Base->SetMobility(EComponentMobility::Movable);
	Base->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	Roof = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Roof"));
	Roof->SetupAttachment(Base);
	if (ConeAsset.Succeeded()) Roof->SetStaticMesh(ConeAsset.Object);
	Roof->SetRelativeScale3D(FVector(0.75f, 0.75f, 0.55f));
	Roof->SetRelativeLocation(FVector(0.f, 0.f, 87.f));
	Roof->SetRelativeRotation(FRotator(0.f, 45.f, 0.f));
	Roof->SetMobility(EComponentMobility::Movable);
	Roof->SetCollisionEnabled(ECollisionEnabled::NoCollision);
}
