# Use Python 3.9-slim as base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Expose the port your Flask app runs on
EXPOSE 5000

# Run the Flask app when the container starts
CMD ["flask", "run", "--host", "0.0.0.0"]
