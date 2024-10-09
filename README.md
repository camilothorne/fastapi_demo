# FastAPI Demo

In this demo, we deploy a vanilla chemical text classifier as a RESTful API endpoint using
the Python [FastAPI](https://fastapi.tiangolo.com/) library.

It takes as input sentences or text chunks and classifies them into `Chemical`, `Biology` or
`Physics` labels, or returns their BPE token stream. As classifier, we rely on fine-tuning a
SciBERT checkpoint from HuggingFace.

## Development instructions

### 1. Set-up Python envinronment

This is the easy part (you can use e.g. Anaconda), follow the next steps:
- Create new Anaconda environment or simple Python3.10+ virtual environment.
- Install the packages using `requirements_docker.txt` file.

### 2. Train models, serialize tokenizer

To fine-tune the model used in the demo, run the Jupyter notebook in `notebooks/`. It shouldn't take
more than a few minutes even on a laptop.

### 3. Run code locally

If you wish to run the code locally, just run the following command in your terminal
```bash
fastapi dev main.py
```

Once deployed you can use either a browser pointed at `http://127.0.0.0:8000/docs` or use `curl` as in
```bash
curl -X 'GET' 'http://localhost:8000/classify/The%20problem%20of%20consciousness%20is%20one%20of%20the%20most%20important%20in%20science.' \
-H 'accept: application/json' | jq
```
or
```bash
curl -X 'GET' 'http://localhost:8000/tokenize/The%20problem%20of%20consciousness%20is%20one%20of%20the%20most%20important%20in%20science.' \
-H 'accept: application/json' | jq
```
Using in all cases the sentence "The problem of consciousness is one of the most important in science." as input.

### 4. Build Docker image

If you have `docker` and `docker-compose` installed on your machine, simply 
run the following command in your terminal (if you are on Linux):
```bash
docker-compose up
```

If you are a MacOSX user, then you'll need to build the image using a special command to ensure the built image is compatible with non ARM hardware:
```bash
docker buildx build --platform linux/amd64 -t <IMAGE_NAME> .
```

Please note that the name (and IP address) of the endpoint may change depending on where you deploy it.
For more information on Docker, please check the project [homepage](https://www.docker.com/).