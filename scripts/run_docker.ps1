$ErrorActionPreference = "Stop"

# Always build from repo root (this script lives in scripts/)
Set-Location (Split-Path $PSScriptRoot -Parent)

$ImageName = "avigpt-api"
$ContainerName = "avigpt-api"

docker build -t $ImageName .
docker run --rm --gpus all -p 8000:8000 --name $ContainerName $ImageName
