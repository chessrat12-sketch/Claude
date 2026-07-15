#include "SimClient.h"
#include "VillageManager.h"
#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "JsonObjectConverter.h"

ASimClient::ASimClient()
{
	PrimaryActorTick.bCanEverTick = false;
}

void ASimClient::BeginPlay()
{
	Super::BeginPlay();
	GetWorldTimerManager().SetTimer(PollTimerHandle, this, &ASimClient::Poll, PollInterval, true, 0.0f);
}

void ASimClient::Poll()
{
	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
	Request->SetURL(ServerUrl + TEXT("/state"));
	Request->SetVerb(TEXT("GET"));
	Request->OnProcessRequestComplete().BindUObject(this, &ASimClient::OnResponseReceived);
	Request->ProcessRequest();
}

void ASimClient::OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully)
{
	if (!bConnectedSuccessfully || !Response.IsValid() || Response->GetResponseCode() != 200)
	{
		UE_LOG(LogTemp, Warning, TEXT("[SimClient] request failed — is viz_server.py running at %s ?"), *ServerUrl);
		return;
	}

	FWorldSnapshot Snapshot;
	if (!FJsonObjectConverter::JsonObjectStringToUStruct(Response->GetContentAsString(), &Snapshot, 0, 0))
	{
		UE_LOG(LogTemp, Warning, TEXT("[SimClient] failed to parse snapshot JSON"));
		return;
	}

	if (Snapshot.tick != LastTick)
	{
		LastTick = Snapshot.tick;
		if (VillageManager)
		{
			VillageManager->ApplySnapshot(Snapshot);
		}
	}
}
