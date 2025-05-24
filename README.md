# EcoSort
AI-powered waste segregation system using computer vision

## Overview
EcoSort is an AI-driven smart waste segregation system that classifies waste into plastic and paper using deep learning and computer vision. The system automates sorting through camera input, ultrasonic sensors, and servo motors.

## Features
- AI-powered classification (EfficientNet-B0)
- Real-time object detection and servo automation
- Background removal and preprocessing using Albumentations
- Hardware integration with ultrasonic sensors and Arduino via Serial

## File Structure
- `src/`: Main code files including prediction, training, and preprocessing
- `annotations/`: Label mappings in JSON format
- `requirements.txt`: Python dependencies

## Installation
```bash
git clone https://github.com/Shwetalray/EcoSort.git
cd EcoSort
pip install -r requirements.txt
