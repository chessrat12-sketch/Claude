#include "AgentActor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/TextRenderComponent.h"
#include "UObject/ConstructorHelpers.h"

AAgentActor::AAgentActor()
{
	PrimaryActorTick.bCanEverTick = false;

	static ConstructorHelpers::FObjectFinder<UStaticMesh> CylinderAsset(
		TEXT("/Engine/BasicShapes/Cylinder.Cylinder"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereAsset(
		TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	static ConstructorHelpers::FObjectFinder<UStaticMesh> CubeAsset(
		TEXT("/Engine/BasicShapes/Cube.Cube"));

	Body = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Body"));
	RootComponent = Body;
	if (CylinderAsset.Succeeded())
	{
		Body->SetStaticMesh(CylinderAsset.Object);
	}
	Body->SetRelativeScale3D(FVector(0.35f, 0.35f, 0.7f));
	Body->SetRelativeLocation(FVector(0.f, 0.f, 35.f));
	Body->SetMobility(EComponentMobility::Movable);
	Body->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	Head = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Head"));
	Head->SetupAttachment(Body);
	if (SphereAsset.Succeeded())
	{
		Head->SetStaticMesh(SphereAsset.Object);
	}
	Head->SetRelativeScale3D(FVector(0.3f));
	Head->SetRelativeLocation(FVector(0.f, 0.f, 65.f));
	Head->SetMobility(EComponentMobility::Movable);
	Head->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	HealthBarBack = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("HealthBarBack"));
	HealthBarBack->SetupAttachment(Body);
	if (CubeAsset.Succeeded())
	{
		HealthBarBack->SetStaticMesh(CubeAsset.Object);
	}
	HealthBarBack->SetRelativeScale3D(FVector(0.5f, 0.06f, 0.06f));
	HealthBarBack->SetRelativeLocation(FVector(0.f, 0.f, 120.f));
	HealthBarBack->SetMobility(EComponentMobility::Movable);
	HealthBarBack->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	HealthBarFill = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("HealthBarFill"));
	HealthBarFill->SetupAttachment(Body);
	if (CubeAsset.Succeeded())
	{
		HealthBarFill->SetStaticMesh(CubeAsset.Object);
	}
	HealthBarFill->SetRelativeScale3D(FVector(0.48f, 0.08f, 0.08f));
	HealthBarFill->SetRelativeLocation(FVector(0.f, -1.f, 120.f));
	HealthBarFill->SetMobility(EComponentMobility::Movable);
	HealthBarFill->SetCollisionEnabled(ECollisionEnabled::NoCollision);

	NameLabel = CreateDefaultSubobject<UTextRenderComponent>(TEXT("NameLabel"));
	NameLabel->SetupAttachment(Body);
	NameLabel->SetRelativeLocation(FVector(0.f, 0.f, 145.f));
	NameLabel->SetWorldSize(24.f);
	NameLabel->SetHorizontalAlignment(EHTA_Center);
	NameLabel->SetTextRenderColor(FColor::White);
	NameLabel->SetMobility(EComponentMobility::Movable);
}

void AAgentActor::SetHealth(int32 Health)
{
	// Scale-only (no colour change): the default engine material doesn't
	// expose a parameter to recolour at runtime without project-specific
	// material setup, so V1 keeps this robust rather than silently doing
	// nothing. Swap in a material with a colour parameter to add that back.
	const float T = FMath::Clamp(Health / 100.f, 0.02f, 1.f);
	HealthBarFill->SetRelativeScale3D(FVector(0.48f * T, 0.08f, 0.08f));
	// Offset so the bar shrinks from the right edge instead of the centre.
	HealthBarFill->SetRelativeLocation(FVector(-0.5f * 0.48f * (1.f - T), -1.f, 120.f));
}

void AAgentActor::SetName(const FString& InName)
{
	NameLabel->SetText(FText::FromString(InName));
}

void AAgentActor::FaceCamera(const FRotator& CameraRotation)
{
	const FRotator Billboard(0.f, CameraRotation.Yaw + 180.f, 0.f);
	NameLabel->SetWorldRotation(Billboard);
}
