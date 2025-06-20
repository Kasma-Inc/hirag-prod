# HiRAG-MCP

## Quick Start

**Option 1: Docker Deployment**
Run the following command to start the container, before that you should install docker and docker-compose.
```bash
git clone https://github.com/Kasma-Inc/hirag-prod
cd hirag-prod
HIRAG_PROD_DIR="." docker compose -p hirag-prod-compose up -d
```
Then use the following command to enter the container:
```bash
docker exec -it $(whoami)_hirag-prod /bin/bash
```
or use VSCode to connect to the container.

Then create the virtual environment and install dependencies using uv as below:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Then create the `.env` file and replace the placeholders with your own values:
```bash
cp .env.example .env
source .env
```

Run the script:
```bash

python main.py
```

**Option 2: Local Deployment**
Make sure uv is installed, if not:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the virtual environment and install dependencies using uv as below:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Then create the `.env` file and replace the placeholders with your own values:
```bash
cp .env.example .env
```

Run the script:
```bash
python main.py
```
