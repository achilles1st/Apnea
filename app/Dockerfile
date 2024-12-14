# Use an appropriate base image with Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app
# Define the command to run the application
CMD ["python", "app.py"]
# Copy the requirements file into the container
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    alsa-utils \
    bluez \
    dbus \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Define the command to run the application
CMD ["python", "app.py"]

## uncomment the following lines to use Gunicorn and remove the CMD line above
## Expose the port (optional but good practice)
#EXPOSE 5000
#
## Define the command to run the application using Gunicorn
#CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app"]
