# 🚀 AI-CDSS 100% Free Production Deployment Guide

This guide walks you through deploying the complete **AI-CDSS** platform for free using:
- **Database:** [Neon Serverless PostgreSQL](https://neon.tech) (Free: 0.5 GB storage, permanent)
- **Backend:** [Hugging Face Spaces](https://huggingface.co/spaces) (Free: **16 GB RAM**, 2 vCPU, full Docker support) OR [Render](https://render.com)
- **Frontend:** [Vercel](https://vercel.com) (Free: global high-speed edge CDN, automated SSL)

---

## 📋 Pre-Flight Checklist

Before you begin, make sure you have:
1. A [GitHub](https://github.com) account.
2. A [Neon](https://neon.tech) account (Sign up with GitHub).
3. A [Hugging Face](https://huggingface.co) account (Sign up with GitHub).
4. A [Vercel](https://vercel.com) account (Sign up with GitHub).

---

## Step 1: Initialize Git & Push to GitHub

From your local machine in the project root directory:

```bash
cd "/Users/animeshtiwari/Downloads/ai-cdss-clean 3/ai-cdss-clean"

# 1. Initialize git
git init -b main

# 2. Stage all project files (safe .gitignore prevents uploading .venv or secrets)
git add .

# 3. Commit
git commit -m "feat: complete AI-CDSS deployment-ready codebase"

# 4. Create a new GitHub repository (public or private) on github.com
# Then link and push:
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
git push -u origin main
```

---

## Step 2: Create Free PostgreSQL Database on Neon

1. Log in to [Neon.tech](https://console.neon.tech).
2. Click **Create Project**:
   - **Project Name:** `ai-cdss-db`
   - **Region:** Choose the region closest to you (e.g., `AWS US East (Ohio)` or `EU Frankfurt`).
3. Click **Create Project**.
4. In your project dashboard, copy the **Connection details (Pooled connection string)**:
   ```text
   postgresql://<user>:<password>@<endpoint>.neon.tech/neondb?sslmode=require
   ```
5. Keep this connection string ready for Step 3.

---

## Step 3: Deploy Backend on Hugging Face Spaces (16 GB Free RAM)

Hugging Face Spaces is recommended because it provides **16 GB RAM for free**, which effortlessly handles PyTorch, 3D MRI segmentation, SHAP, and ECG ResNet without crashing.

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Configure your Space:
   - **Space Name:** `ai-cdss-backend`
   - **License:** `apache-2.0` (or MIT)
   - **Select Space SDK:** Choose **Docker** 🐳 -> **Blank**
   - **Space Hardware:** Select **CPU basic (2 vCPU · 16 GB RAM · Free)**
   - **Visibility:** Public (or Private)
3. Click **Create Space**.
4. In the Space **Settings** tab -> scroll to **Variables and Secrets**:
   - Add Secret: `DATABASE_URL` = `<your-neon-postgres-connection-string>`
   - Add Secret: `JWT_SECRET_KEY` = `<generate a 32-char random string>`
   - Add Variable: `ENVIRONMENT` = `production`
   - Add Variable: `DEBUG` = `false`
   - Add Variable: `ALLOWED_ORIGINS` = `*` *(or your Vercel frontend URL once deployed)*
5. Push your code to your Hugging Face Space repository:
   ```bash
   # Add Hugging Face Space as a remote
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/ai-cdss-backend

   # Push the main branch to Hugging Face
   git push hf main:main
   ```
6. Hugging Face will automatically detect `docker/Dockerfile.backend` (or move it to the root `Dockerfile` if prompted).
   *Note:* In the root directory, create a 1-line symlink or copy if needed:
   ```bash
   cp docker/Dockerfile.backend Dockerfile
   git add Dockerfile && git commit -m "add root Dockerfile for HF Space" && git push hf main:main
   ```
7. Once the build completes, copy your Space URL:
   ```text
   https://<YOUR_HF_USERNAME>-ai-cdss-backend.hf.space
   ```
   Test it in your browser: `https://<YOUR_HF_USERNAME>-ai-cdss-backend.hf.space/docs`

---

## Step 4: Deploy Frontend on Vercel

1. Go to [vercel.com/new](https://vercel.com/new).
2. Select your GitHub repository: `<YOUR_REPO_NAME>`.
3. Configure Project Settings:
   - **Framework Preset:** `Vite`
   - **Root Directory:** Click **Edit** and select `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
   - **Install Command:** `npm install`
4. Expand **Environment Variables**:
   - **Key:** `VITE_API_BASE_URL`
   - **Value:** `https://<YOUR_HF_USERNAME>-ai-cdss-backend.hf.space/api`
5. Click **Deploy**.
6. In ~60 seconds, your frontend will be live at `https://your-project.vercel.app`!

---

## Step 5: Final Security & CORS Handshake

Once you have your Vercel URL (e.g. `https://ai-cdss.vercel.app`):
1. Go back to your Hugging Face Space **Settings** -> **Variables and Secrets**.
2. Update `ALLOWED_ORIGINS`:
   ```text
   https://ai-cdss.vercel.app,http://localhost:5173
   ```
3. Restart the Space.

Your full AI-CDSS system is now completely deployed, interconnected, secure, and running 100% free!
