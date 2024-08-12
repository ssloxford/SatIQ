#!/bin/bash

db_path=../data
storage_path=$db_path/completed

# Stop the pipeline gracefully
docker-compose down

# Create a timestamp
timestamp=$(date +%Y-%m-%d)

# Rename the database file with the timestamp
mv "$db_path/db.sqlite3" "$db_path/db-$timestamp.sqlite3"

# Bring the pipeline back online
docker-compose up -d

# Move the database file to the long-term storage location
mv "$db_path/db-$timestamp.sqlite3" "$storage_path/db-$timestamp.sqlite3"
