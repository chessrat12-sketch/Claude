// Polls the Python viz server for world snapshots and hands them to a
// AVillageManager. Nothing here decides agent behaviour — it is a pure
// client of the simulation running in Python (the single source of truth).
//
// Requires your project's Build.cs to list "HTTP", "Json", "JsonUtilities" in
// PublicDependencyModuleNames — see unreal/README_UNREAL.md.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Interfaces/IHttpRequest.h"
#include "SimClient.generated.h"

class AVillageManager;

UCLASS()
class ASimClient : public AActor
{
	GENERATED_BODY()

public:
	ASimClient();

	UPROPERTY(EditAnywhere, Category = "Emergent Civilization")
	FString ServerUrl = TEXT("http://localhost:8000");

	UPROPERTY(EditAnywhere, Category = "Emergent Civilization")
	float PollInterval = 0.3f;

	UPROPERTY(EditAnywhere, Category = "Emergent Civilization")
	AVillageManager* VillageManager;

protected:
	virtual void BeginPlay() override;

private:
	FTimerHandle PollTimerHandle;
	int32 LastTick = -1;

	void Poll();
	void OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully);
};
