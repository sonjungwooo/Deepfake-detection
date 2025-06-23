# DeepFake-detection
> Made for Startup competition

## 1. Business Overview

### 1-1. Motivation for the Startup Idea

The rapid advancement of AI technology has revolutionized how information is produced and delivered. However, it has also given rise to unintended consequences such as “deepfakes.” This technology can realistically synthesize a person’s face and voice, creating false information that can damage reputations and erode public trust.

In particular, teenagers and the general public often lack the means to distinguish deepfakes, making them vulnerable to misinformation. To address this societal issue through technology, we propose a mobile-friendly app that enables anyone to easily detect deepfake videos. The goal is to enhance information reliability and promote digital ethics.

⸻

### 1-2. Core Content of the Startup Idea

This app utilizes a CNN-based deep learning model to analyze various visual features in videos, such as facial expressions, movements, skin textures, and frame transitions. Users simply upload a video, and the AI automatically determines its authenticity, presenting the result through an intuitive graphical interface.

• Core Technologies:
• CNN deep learning model built with TensorFlow
• Frame extraction and preprocessing via OpenCV
• Flask-based API server linked to mobile application

• Key Features:
• Lightweight AI optimized for mobile environments
• One-click automatic analysis
• Visualized results for deepfake detection

• Scalability:
• Potential partnerships with schools, media outlets, and social media platforms
• API availability for integration with external platforms

⸻

## 2. Market Analysis

### 2-1. Current Market Landscape

Currently, deepfake detection technologies are being researched and offered by institutions like Microsoft and MIT. However, these tools are often designed for experts and are not easily accessible to the general public.

There are virtually no mobile-friendly deepfake detection services, despite a clear demand from non-expert users.

• Market Gap: Teenagers, general public, educators
• Unmet Needs: User-friendly, mobile-compatible solutions
• Potential Collaborations: National Police Agency, KISA, Korea Communications Commission, etc.
• Intellectual Property: Patent applications planned for AI models, UI/UX designs

⸻

### 2-2. Business Feasibility & Sales Strategy

• Funding Sources:
• Government programs (K-Startup, Youth Startup Academy, etc.)
• Startup competition prizes
• Crowdfunding and early-stage investors

• Revenue Model:
• Core features available for free (ad-supported)
• Paid premium services (advanced detection, history logging, etc.)
• Separate licensing for educational institutions (admin dashboard access)

• Market Entry Strategy:
• Pilot implementation in schools → Promotion via social media
• Viral short-form content on TikTok and Instagram
• Collaborations with influencers to build early user base

• Technical Advantage:
• Lightweight model capable of real-time analysis
• Self-improving AI through automatic updates
• Higher reliability and user experience than open-source alternatives

## Program Structure
```
deepfake_detection_program/
├── model.py             # 1. 모델 정의: Xception 기반 딥러닝 모델 생성
├── preprocessing.py     # 2. 얼굴 추출 함수: 비디오에서 얼굴 영역 추출
├── detector.py          # 3. 딥페이크 탐지 함수: 얼굴을 처리해 모델 예측 및 확률 계산
└── main.py
```

⸻

## Website
```
my-webapp/
├── app.py         ← Flask 서버
├── templates/
│   └── index.html ← 프론트
├── static/
│   ├── style.css
│   └── script.js
```
