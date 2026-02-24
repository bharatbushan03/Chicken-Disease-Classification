---
title: Chicken Disease Classification
emoji: 🐔
colorFrom: orange
colorTo: red
sdk: docker
pinned: false
app_port: 7860
---

# Chicken Disease Classification

A deep learning web app that classifies chicken diseases from images using a VGG16-based CNN model.

## Usage

Upload a chicken image and the model will predict whether it shows signs of disease.

## API

- `GET /` — Web interface
- `POST /predict` — JSON body: `{ "image": "<base64>" }` → returns `{ "label": "...", "confidence": 0.95 }`