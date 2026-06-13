# SignBridge

## Real-Time ASL Translation Assistant

SignBridge is an Edge AI-powered accessibility platform that translates American Sign Language (ASL) into natural English text and speech in real time.

The project combines computer vision, transformer-based gesture recognition, and large language models to bridge communication between Deaf and hearing individuals. By leveraging on-device AI for gesture recognition and cloud-based language understanding, SignBridge provides fast, accurate, and privacy-conscious translation.

---

## Problem Statement

Millions of Deaf and hard-of-hearing individuals rely on sign language as their primary means of communication. However, communication barriers still exist when interacting with people who do not understand sign language.

Existing solutions often suffer from:

* Limited vocabulary
* High latency
* Expensive hardware requirements
* Lack of contextual understanding
* Poor sentence construction
* No conversational assistance

SignBridge aims to address these challenges through a combination of Edge AI and Large Language Models.

---

## Project Goals

### Primary Goals

* Real-time ASL recognition using a laptop webcam
* On-device gesture recognition using Edge AI
* Translation of recognized signs into English
* Speech synthesis of translated text
* Context-aware sentence generation

### Advanced Goals

* Facial expression understanding
* Conversational context awareness
* Next-word prediction
* Continuous sign recognition
* Personalized translation experience

---

## System Architecture

```text
Laptop Webcam
      │
      ▼
MediaPipe Holistic
(Hands + Face + Pose)
      │
      ▼
Transformer Recognition Model
      │
      ▼
ASL Word Prediction
      │
      ▼
Spring Boot Backend
      │
 ┌────┴─────┐
 ▼          ▼
OpenAI      PostgreSQL
API         Database
 │
 ▼
Sentence Generation
 │
 ▼
Text-to-Speech
 │
 ▼
Natural Language Output
```

---

## Technology Stack

### Frontend

* React
* TypeScript
* Vite
* WebRTC
* WebSocket

### Backend

* Java 21
* Spring Boot
* Spring Data JPA
* PostgreSQL
* Maven

### AI Service

* Python
* TensorFlow Lite
* MediaPipe
* OpenCV
* FastAPI

### AI Models

* Transformer Encoder
* MediaPipe Holistic Tracking
* OpenAI API

### DevOps & Collaboration

* GitHub
* GitHub Projects
* GitHub Issues

---

## Core Components

### 1. Computer Vision Layer

The webcam feed is processed using MediaPipe Holistic to extract:

* Hand landmarks
* Facial landmarks
* Pose landmarks

This converts raw video into structured landmark data suitable for machine learning.

### 2. Sign Recognition Layer

A Transformer-based recognition model processes landmark sequences and predicts ASL signs.

Features:

* Real-time inference
* Edge AI deployment
* TensorFlow Lite optimization
* Low latency

### 3. Language Understanding Layer

Recognized signs are passed to OpenAI models to:

* Construct grammatically correct English
* Understand context
* Predict next words
* Improve sentence quality

### 4. Speech Layer

Translated sentences are converted into natural speech through Text-to-Speech services.

---

## Repository Structure

```text
signbridge/
│
├── frontend/
│   └── React Application
│
├── backend/
│   └── Spring Boot APIs
│
├── ai-service/
│   ├── MediaPipe
│   ├── Transformer
│   ├── Training
│   └── Inference
│
├── docs/
│   ├── architecture.md
│   ├── api-spec.md
│   └── team-responsibilities.md
│
├── datasets/
│
├── presentations/
│
└── README.md
```

---


## Example Workflow

Input:

```text
STORE
I
GO
TOMORROW
```

Recognition Output:

```text
STORE I GO TOMORROW
```

OpenAI Translation:

```text
I am going to the store tomorrow.
```

Speech Output:

```text
"I am going to the store tomorrow."
```

---

## Future Enhancements

* Multi-language support
* Mobile application
* Offline language model
* Personalized sign profiles
* Bidirectional communication
* Real-time conversation mode

---

## License

This project is being developed as part of an AI accessibility initiative focused on improving communication and inclusion through Edge AI and Large Language Models.

---

## Contributors

Team SignBridge

Building accessible communication through AI.
