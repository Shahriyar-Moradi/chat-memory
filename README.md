# 🚀 Memory Chatbot API - Backend Deployment

Deploy your FastAPI backend to Google Cloud Run with Docker.

## 📋 Prerequisites

1. **Google Cloud Platform Account**
2. **Google Cloud CLI** installed and configured
3. **Docker** installed and running
4. **Project with billing enabled**

## 🔧 Setup

### 1. Install Google Cloud CLI
```bash
# macOS
brew install --cask google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### 2. Authenticate with Google Cloud
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Enable Required APIs
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## 🚀 Deployment Options

### Option 1: Quick Deployment Script

```bash
cd backend
./deploy.sh YOUR_PROJECT_ID
```

### Option 2: Manual Deployment

```bash
cd backend

# Build Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/memory-chatbot-api .

# Push to Container Registry
docker push gcr.io/YOUR_PROJECT_ID/memory-chatbot-api

# Deploy to Cloud Run
gcloud run deploy memory-chatbot-api \
    --image=gcr.io/YOUR_PROJECT_ID/memory-chatbot-api \
    --platform=managed \
    --region=us-central1 \
    --allow-unauthenticated \
    --port=8080 \
    --memory=1Gi \
    --cpu=1
```

### Option 3: Automated CI/CD with Cloud Build

```bash
# Submit build to Cloud Build
gcloud builds submit --config=cloudbuild.yaml ../
```

## 🔐 Environment Variables

After deployment, set these environment variables in Cloud Run:

```bash
gcloud run services update memory-chatbot-api \
    --region=us-central1 \
    --set-env-vars="OPENAI_API_KEY=your_openai_key,PINECONE_API_KEY=your_pinecone_key,MEMORY_INDEX_NAME=chatbot-memory"
```

Or set them in the Google Cloud Console:
- Go to Cloud Run → memory-chatbot-api → Edit & Deploy New Revision
- Add environment variables:
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `PINECONE_API_KEY`: Your Pinecone API key
  - `MEMORY_INDEX_NAME`: chatbot-memory (or your preferred index name)

## 📡 Testing Your Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe memory-chatbot-api --region=us-central1 --format='value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health

# Test API info
curl $SERVICE_URL/info

# View API documentation
open $SERVICE_URL/docs
```

## 📊 Monitoring & Logs

### View Logs
```bash
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=memory-chatbot-api" --limit=50
```

### Monitor Performance
- Go to Google Cloud Console → Cloud Run → memory-chatbot-api
- View metrics, logs, and performance data

## 🔧 Configuration

### Docker Configuration
- **Base Image**: `python:3.10-slim`
- **Port**: 8080 (Cloud Run standard)
- **Memory**: 1Gi
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds
- **Max Instances**: 10

### Security Features
- Non-root user in container
- Minimal system dependencies
- Health checks enabled
- Environment variables for secrets

## 🛠️ Troubleshooting

### Common Issues

1. **Build Fails**
   ```bash
   # Check logs
   gcloud builds log $(gcloud builds list --limit=1 --format='value(id)')
   ```

2. **Service Won't Start**
   ```bash
   # Check service logs
   gcloud logs read "resource.type=cloud_run_revision" --limit=20
   ```

3. **Environment Variables Missing**
   ```bash
   # List current environment variables
   gcloud run services describe memory-chatbot-api --region=us-central1 --format='export'
   ```

### Performance Tuning

```bash
# Update service configuration
gcloud run services update memory-chatbot-api \
    --region=us-central1 \
    --memory=2Gi \
    --cpu=2 \
    --concurrency=100 \
    --max-instances=20
```

## 📋 Files Overview

- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
- `.dockerignore` - Files to exclude from build
- `deploy.sh` - Quick deployment script
- `cloudbuild.yaml` - CI/CD configuration
- `README.md` - This documentation

## 🔗 Useful Links

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/) 