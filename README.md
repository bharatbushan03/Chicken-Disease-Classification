# Chicken Disease Classification

A Flask application that classifies chicken images as **Coccidiosis** or **Healthy** using a Convolutional Neural Network (CNN). This project is designed for both local development and easy deployment on platforms like Render.

## Features
- **Browser-based UI:** Simple interface for image upload and prediction.
- **RESTful API:** JSON endpoints for integration with other services.
- **Base64 Support:** Accepts raw or data-URL prefixed base64 image data.
- **ML Pipeline:** Built-in training pipeline (Data Ingestion, Model Preparation, Training, Evaluation).

## Project Structure
- `app.py`: Flask application entry point.
- `src/CNNClassifier`: Core logic for the CNN classifier.
- `templates/`: HTML templates for the web interface.
- `params.yaml`: Configuration parameters for the ML pipeline.
- `requirements.txt`: Python dependencies.

## Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/bharatbushan03/Chicken-Disease-Classification.git
   cd Chicken-Disease-Classification
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/macOS
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```
5. Open [http://localhost:5000](http://localhost:5000) in your browser.

## API Endpoints
- **GET `/`**: Home page.
- **GET `/health`**: Health check to verify service and model status.
- **POST `/predict`**:
    - **Body**: `{"image": "data:image/jpeg;base64,..."}`
    - **Response**: `{"label": "Healthy", "confidence": 0.95}`
- **POST `/train`**: Triggers the training pipeline (requires `ENABLE_TRAINING=true`).

## Configuration
- `MODEL_PATH`: Path to the trained model file (default: `model.keras`).
- `ENABLE_TRAINING`: Set to `true` to enable the training endpoint.
- `PORT`: The port on which the server runs (default: `5000`).

## Deployment
This project is configured for one-click deployment on [Render](https://render.com/) using the provided `render.yaml`. Ensure `model.keras` is committed to your repository for immediate use.