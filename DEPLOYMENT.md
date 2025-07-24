# 🚀 Deployment Guide

Your House Price Predictor is now ready for deployment! Here are several options to deploy your application.

## 📍 **GitHub Repository**
✅ **Already deployed to**: https://github.com/lindaassefa/house-price-predictor

## 🎯 **Deployment Options**

### 1. **Render (Recommended - Free Tier)**
**Best for**: Quick deployment with free tier

**Steps:**
1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub account
4. Select the `house-price-predictor` repository
5. Configure:
   - **Name**: `house-price-predictor-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python api.py`
6. Click "Create Web Service"

**Features:**
- ✅ Free tier available
- ✅ Automatic deployments from GitHub
- ✅ Custom domain support
- ✅ SSL certificates included

---

### 2. **Railway (Recommended - Free Tier)**
**Best for**: Simple deployment with good performance

**Steps:**
1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select the `house-price-predictor` repository
4. Railway will automatically detect the configuration
5. Deploy!

**Features:**
- ✅ Free tier available
- ✅ Automatic deployments
- ✅ Built-in monitoring
- ✅ Easy scaling

---

### 3. **Heroku (Paid)**
**Best for**: Enterprise applications

**Steps:**
1. Install Heroku CLI: `brew install heroku/brew/heroku`
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`
5. Open: `heroku open`

**Features:**
- ✅ Professional hosting
- ✅ Advanced monitoring
- ✅ Add-ons ecosystem
- ✅ Team collaboration

---

### 4. **Google Cloud Platform**
**Best for**: Scalable enterprise solutions

**Steps:**
1. Install Google Cloud SDK
2. Create a new project
3. Enable Cloud Run API
4. Deploy using: `gcloud run deploy`

---

### 5. **AWS (Amazon Web Services)**
**Best for**: Enterprise applications with AWS ecosystem

**Steps:**
1. Create AWS account
2. Use AWS Elastic Beanstalk
3. Upload your application
4. Configure environment

---

## 🔧 **Local Testing Before Deployment**

Test your application locally:

```bash
# Test API
python api.py

# Test in another terminal
curl http://localhost:5001/health
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "State": "CA",
    "City": "Los Angeles",
    "Street": "123 Main St",
    "Zipcode": 90210,
    "Bedroom": 3,
    "Bathroom": 2.5,
    "Area": 2000.0,
    "PPSq": 500.0,
    "LotArea": 500.0,
    "MarketEstimate": 1000000.0,
    "RentEstimate": 3000.0
  }'
```

## 📊 **API Endpoints**

Once deployed, your API will have these endpoints:

- **Health Check**: `GET /health`
- **Single Prediction**: `POST /predict`
- **Batch Prediction**: `POST /predict/batch`
- **Data Statistics**: `GET /data/stats`
- **Cities by State**: `GET /data/cities/<state>`

## 🌐 **Web Interface**

To deploy the Streamlit web interface separately:

1. Create a new service on your chosen platform
2. Use start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. Add to requirements.txt if not already there

## 🔒 **Environment Variables**

Your application uses these environment variables:
- `PORT`: Server port (default: 5001)
- `DEBUG`: Debug mode (default: False)

## 📈 **Monitoring & Maintenance**

### Health Monitoring
- Use the `/health` endpoint for monitoring
- Set up alerts for response time and availability

### Logs
- Check application logs regularly
- Monitor error rates and performance

### Updates
- Push changes to GitHub
- Most platforms auto-deploy from main branch

## 🆘 **Troubleshooting**

### Common Issues:

1. **Port Issues**
   - Ensure `PORT` environment variable is set
   - Check if port is available

2. **Model Loading**
   - Verify `house_price_model.pkl` is in repository
   - Check file permissions

3. **Dependencies**
   - Ensure all packages in `requirements.txt`
   - Check Python version compatibility

4. **Memory Issues**
   - Model file is ~38MB
   - Ensure adequate memory allocation

## 🎉 **Success!**

Once deployed, your House Price Predictor will be available at:
- **API**: `https://your-app-name.onrender.com` (or your chosen platform)
- **Health Check**: `https://your-app-name.onrender.com/health`

Share your deployed API with others and start making predictions! 🏠✨ 