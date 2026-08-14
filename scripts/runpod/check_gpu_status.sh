#!/bin/bash

set -o allexport
source .env
set +o allexport

runpodctl gpu list \
    | jq -r '.[] 
      | select(.secureCloud==true and .available==true) 
      | select(any(.dataCenterAvailability[]; .dataCenterId=="EUR-IS-1" and .stockStatus!="none"))
      | "\(.displayName),\(.gpuId),\(.memoryInGb)GB,$\(.securePricePerHr)"' \
    | column -t -s ','