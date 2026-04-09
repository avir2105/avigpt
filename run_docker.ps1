$ErrorActionPreference = "Stop"

$ImageName = "avigpt-api"
$ContainerName = "avigpt-api"

docker build -t $ImageName .
docker run --rm --gpus all -p 8000:8000 --name $ContainerName $ImageName