curl --location 'http://localhost:8000/v2/models/nvidia-triton-llm-streaming/versions/1/generate_stream'           --header 'Accept: text/event-stream'           --data '{
  "parameters" :
    {
      "TEXT" : "How to make a omellete"
    }
}'
