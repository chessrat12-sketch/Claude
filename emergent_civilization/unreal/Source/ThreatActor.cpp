#include "ThreatActor.h"
#include "Components/StaticMeshComponent.h"
#include "UObject/ConstructorHelpers.h"

AThreatActor::AThreatActor()
{
	PrimaryActorTick.bCanEverTick = false;

	static ConstructorHelpers::FObjectFinder<UStaticMesh> CylinderAsset(
		TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereAsset(
		TEXT("/Engine/BasicShapes/Sphere.Sphere"));

	Body = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Body"));
	RootComponent = Body;
	if (CylinderAsset.Succeeded()) Body->SetStaticMesh(CylinderAsset.Object);
	Body->SetRelativeScale3D(FVector(0.25f, 0.25f, 0.4f));
	Body->SetRelativeLocation(FVector(0.f, 0.f, 20.f));
	Body->SetMobility(EComponentMobility::Movable);
	Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	Head = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Head"));
	Head->SetupAttachment(Body);
	if (SphereAsset.Succeeded()) Head->SetStaticMesh(SphereAsset.Object);
	Head->SetRelativeScale3D(FVector(0.2f));
	Head->SetRelativeLocation(FVector(15.f, 0.f, 35.f));
	Head->SetMobility(EComponentMobility::Movable);
	Head->SetCollisionEnabled(ECollisionEnabled::NoCollision);
}
