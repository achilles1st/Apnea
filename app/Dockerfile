FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies and cron
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libportaudio2 \
    portaudio19-dev \
    alsa-utils \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Copy the crontab file to the cron directory
COPY crontab /etc/cron.d/my-cron

# Set the appropriate permissions for the crontab file
RUN chmod 0644 /etc/cron.d/my-cron

# Register the crontab
RUN crontab /etc/cron.d/my-cron

# Create a log file for cron logs
RUN touch /var/log/cron.log

# Expose the Flask app port
EXPOSE 5000

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
