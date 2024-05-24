# Deploy Zephr Using Streaming(SSE) with Nvidia Triton for Python Backend 


In the world of machine learning and AI, real-time data processing and inference can significantly enhance the responsiveness and efficiency of applications. Server-Sent Events (SSE) offer a straightforward method to push updates to a client over HTTP, making it an ideal choice for real-time data streaming. Combined with NVIDIA Triton, an open-source inference server that supports multiple machine learning frameworks, developers can deploy and manage scalable AI applications efficiently. This blog will guide you through integrating SSE with NVIDIA Triton using a Python backend.

**What is SSE?**

Server-Sent Events (SSE) is a standard describing how servers can initiate data transmission towards browser clients once an initial client connection has been established. It’s particularly useful for creating a one-way communication channel from the server to the client, such as for real-time notifications, live updates, and streaming data.

**What is NVIDIA Triton?**

NVIDIA Triton Inference Server provides a robust, scalable serving system for deploying machine learning models from any framework (TensorFlow, PyTorch, ONNX Runtime, etc.) over GPUs and CPUs in production environments. Triton simplifies deploying multiple models from different frameworks, optimizing GPU utilization and integrating with Kubernetes.

**How to use the Model?**

* Create a folder model_repo
* Clone the repo inside the model_repo
* Run the docker command
```
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ~/model_repo:/models nvcr.io/nvidia/tritonserver:23.11-py3 tritonserver --model-repository=/models --model-control-mode=explicit
```
* Install the dependencies inside docker with docker exec 
```
pip install "autoawq==0.1.8"
pip install "torch==2.1.2"
```
* Load the model explicitely
curl --location --request POST 'http://localhost:8000/v2/repository/models/nvidia-triton-llm-streaming/load'
* Run 'test.sh' to call inference 



