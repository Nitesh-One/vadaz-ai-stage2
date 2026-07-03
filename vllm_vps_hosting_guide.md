# Hosting a Model on a VPS using vLLM

This guide provides a step-by-step walkthrough for deploying a model (such as `Qwen/Qwen2.5-7B-Instruct`) on a Virtual Private Server (VPS) with an NVIDIA GPU using **vLLM**, which is one of the fastest and most memory-efficient engines for LLM serving.

---

## 1. Hardware & OS Requirements

To host a model like **Qwen 2.5 7B** (in 16-bit or 8-bit precision), your VPS should ideally meet these minimum requirements:
*   **Operating System**: Ubuntu 22.04 LTS (strongly recommended)
*   **GPU**: 1x NVIDIA GPU with at least **16 GB to 24 GB VRAM** (e.g., RTX 3090, RTX 4090, A10G, or L4).
*   **RAM**: At least 32 GB System RAM.
*   **Storage**: 50 GB+ NVMe SSD (to store base model weights, which occupy ~15GB for a 7B model).

---

## 2. NVIDIA Drivers & CUDA Installation

Ensure that your GPU drivers and CUDA toolkit are installed and correctly configured on the VPS.

```bash
# Update package database
sudo apt update && sudo apt upgrade -y

# Install NVIDIA Drivers (if not already installed)
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Verify installation (requires reboot if newly installed)
nvidia-smi
```

---

## 3. Docker & NVIDIA Container Toolkit Setup

Running vLLM via Docker is highly recommended because it comes with pre-compiled CUDA, PyTorch, and xFormers dependencies, avoiding dependency conflicts.

### Install Docker:
```bash
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Install NVIDIA Container Toolkit:
This allows Docker containers to access your physical GPU.
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker to apply changes
sudo systemctl restart docker
```

---

## 4. Run the vLLM Serving Container

Create a local cache directory to store Hugging Face model weights so they aren't lost if the container is restarted:

```bash
mkdir -p /home/ubuntu/.cache/huggingface
```

Now, launch the vLLM server container. Replace `Qwen/Qwen2.5-7B-Instruct` with your model of choice.

```bash
docker run -d --gpus all \
  --name vllm-server \
  -p 8000:8000 \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
```

### Crucial Launch Flags:
*   `--gpus all`: Passes all available GPUs to the Docker container.
*   `-p 8000:8000`: Exposes the container port 8000 to the VPS.
*   `--model`: Name of the model on Hugging Face (it will download automatically on first run).
*   `--gpu-memory-utilization 0.90`: Allocates up to 90% of your VRAM to vLLM KV Cache (reduces out-of-memory errors).
*   `--max-model-len 4096`: Limits the max context size to fit smaller VRAM cards.
*   `--tensor-parallel-size <n>`: Add this if you have multiple GPUs (e.g., `--tensor-parallel-size 2` for 2 GPUs).

Monitor download and startup logs:
```bash
docker logs -f vllm-server
```
Wait until you see: `INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`

---

## 5. Verify the Server Endpoint

Since vLLM exposes an **OpenAI-compatible API**, you can verify it immediately using `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

---

## 6. Secure Hosting: Nginx + SSL + Authentication

It is unsafe to expose port 8000 directly to the internet. We will use **Nginx** as a reverse proxy, **Certbot** for SSL (HTTPS), and configure an API token.

### Step A: Configure vLLM with an API Key
Re-run the Docker container with the `--api-key` parameter to secure the endpoints:

```bash
docker stop vllm-server && docker rm vllm-server

docker run -d --gpus all \
  --name vllm-server \
  -p 127.0.0.1:8000:8000 \
  -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --api-key "your-super-secret-api-key" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
```
*(Notice we changed `-p 8000:8000` to `-p 127.0.0.1:8000:8000` so only localhost can hit the port directly).*

### Step B: Setup Nginx
Install Nginx:
```bash
sudo apt install nginx -y
```

Create a new Nginx block `/etc/nginx/sites-available/vllm`:
```nginx
server {
    listen 80;
    server_name your-domain.com; # Replace with your domain name or IP

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Disable buffering for streaming responses (Server-Sent Events)
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 600s;
    }
}
```

Enable the configuration and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/vllm /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### Step C: Secure with SSL (Let's Encrypt)
Install Certbot for automated SSL certificates:
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```
Follow the interactive prompts to enable HTTP to HTTPS redirect.

---

## 7. Connecting to the VPS Server

Once set up, you can configure your Python applications (e.g., `chat_generator.py` or `quality_tester.py`) to call your custom endpoint by updating the `.env` settings:

```env
OPENAI_BASE_URL=https://your-domain.com/v1
OPENAI_API_KEY=your-super-secret-api-key
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
```
