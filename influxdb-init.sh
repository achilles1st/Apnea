#!/bin/bash

# Wait for InfluxDB to fully start
sleep 10

# Set variables
INFLUXDB_URL="http://localhost:8086"
ORG="my-org"
BUCKET="my-bucket"
USER="admin"
PASSWORD="password"

# Create the initial organization, bucket, and user
influx setup --skip-verify --bucket $BUCKET --org $ORG --username $USER --password $PASSWORD --retention 0 --force

# Generate a token
TOKEN=$(influx auth create --description "admin-token" --write-buckets --read-buckets --json | jq -r '.token')

# Save the token to a file for later use
echo "INFLUX_TOKEN=$TOKEN" > /var/lib/influxdb/influxdb.token
echo "Token created and saved successfully."
