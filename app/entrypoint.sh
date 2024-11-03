#!/bin/bash

# Start cron in the background
cron &

# Start the Flask app
python app.py
