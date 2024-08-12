#!/bin/bash

docker-compose up -d

while true; do
  # Wait for 24 hours
  sleep 24h

  # Run restart.sh script
  ./restart.sh
done

