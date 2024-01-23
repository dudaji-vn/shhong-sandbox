# Installation
```
> python load.py # load model.pt
> cp model.pt ./model_repository/resnet50/

> cd docker && docker build -t ai-chip-client ./
```

https://pytorch.org/TensorRT/tutorials/serving_torch_tensorrt_with_triton.html

# Run triton server
```
docker run --gpus '"device=0"' --cpus 16 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 --net=host \
  --name ai-chip-triton-server -v /home/dudaji/shhong/model_repository:/models \
  nvcr.io/nvidia/tritonserver:22.04-py3 tritonserver --model-repository=/models
```

# Run argos servers
```
# helper
docker run -it --rm --gpus '"device=0"' --cpus 64  -p 4999:5000  \
  --name ai-chip-client-4999 \
  -v ${PWD}/:/work ai-chip-client bash
> cd /work && python app.py

# inference
docker run -it --rm --gpus '"device=0"' --cpus 16  -p 5000:5000  \
  --name ai-chip-client-5000 \
  -v ${PWD}/:/work ai-chip-client bash
> cd /work && python app.py
```

# Test 
```
docker run -it --rm --gpus '"device=0"' --cpus 2  --net=host \
  --name ai-chip-client-cli \
  -v ${PWD}/:/work ai-chip-client bash
> cd /work && python benchmark.py
```




