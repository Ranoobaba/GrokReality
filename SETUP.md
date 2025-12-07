# Quick Setup Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Get X OAuth Credentials

1. Visit https://developer.twitter.com/en/portal/dashboard
2. Create a new app
3. Enable OAuth 2.0
4. Set callback URL to: `http://localhost:5000/callback`
5. Copy your Client ID and Client Secret

## 3. Set Environment Variables

```bash
export XAI_API_KEY='your-xai-api-key'
export X_CLIENT_ID='your-x-client-id'
export X_CLIENT_SECRET='your-x-client-secret'
export X_REDIRECT_URI='http://localhost:5000/callback'
```

## 4. Run the App

```bash
python app.py
```

## 5. Open in Browser

Navigate to: http://localhost:5000

## Troubleshooting

- **Microphone not working**: Make sure to allow microphone access in your browser
- **OAuth errors**: Verify your callback URL matches exactly in Twitter Developer Portal
- **API errors**: Check that your XAI_API_KEY is valid and has proper permissions

