# Build the Docker image
docker build -t pdf-conversion-service .

# Run the Docker container
docker run --gpus all -p 8003:8003 pdf-conversion-service