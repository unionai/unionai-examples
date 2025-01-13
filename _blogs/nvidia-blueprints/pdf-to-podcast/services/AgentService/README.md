 docker build -t agent-service .


 docker run -p 8964:8964 -e NVIDIA_API_KEY=$NVIDIA_API_KEY  agent-service

 Agent Service in Langgraph
 
