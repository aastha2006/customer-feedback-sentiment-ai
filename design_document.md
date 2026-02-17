
# Design Document - Customer Feedback Analysis System

## 1. System Architecture

The system follows a modular architecture designed for scalability and maintainability.

### Components

1.  **Data Ingestion Layer (`src/data_ingestion.py`)**: Responsible for loading data from CSV files or generating synthetic data for testing. It handles basic validation and cleaning.
2.  **Preprocessing Layer (`src/preprocessing.py`)**: Standardizes text data (lowercasing, special character removal) and converts it into numerical vectors using TF-IDF. This layer is consistent across training and inference.
3.  **Model Layer (`src/train.py`, `src/predict.py`)**:
    -   **Training**: Splits data, trains a Logistic Regression model, and saves artifacts using `joblib`.
    -   **Inference**: Loads artifacts as a Singleton to ensure efficiency and exposes a simple `predict` interface.
4.  **API Layer (`api/main.py`)**: A FastAPI application that serves the model. It handles request validation using Pydantic, logging, and error handling.
5.  **Infrastructure**: Docker is used for containerization, ensuring valid environments.

## 2. Model Selection Reasoning

**Logistic Regression with TF-IDF** was chosen for the initial version because:
-   **Baseline Performance**: It provides a strong baseline for text classification tasks.
-   **Interpretability**: Feature importance is easily extracting.
-   **Speed**: Training and inference are extremely fast (<10ms per prediction), meeting the <200ms requirement.
-   **Resource Efficiency**: Low memory footprint compared to Transformer models (BERT, etc.).

## 3. Scalability Considerations

-   **Stateless API**: The FastAPI app is stateless, allowing horizontal scaling by deploying multiple replicas behind a load balancer (e.g., Kubernetes Ingress or Nginx).
-   **Singleton Model Loading**: The model is loaded once per worker process, minimizing overhead.
-   **Batch Prediction**: The `/batch_predict` endpoint allows processing multiple records in a single request, reducing network overhead.
-   **Asynchronous Support**: FastAPI's async capabilities allow handling concurrent requests efficiently.

## 4. Deployment Strategy

-   **Docker**: The application is containerized, making it deployable on any platform supporting Docker (AWS ECS, Google Cloud Run, Azure Container Instances).
-   **CI/CD**: A pipeline (implied by Makefile) can automate testing and building images.
-   **Monitoring**: Logs are written to `logs/app.log` and stdout, which can be aggregated by tools like ELK Stack or Datadog.

## 5. Future Improvements

-   **Advanced Models**: Experiment with BERT/RoBERTa for potentially higher accuracy on complex sentences.
-   **Model Monitoring**: Integrate Prometheus/Grafana to track prediction distributions and detect drift.
-   **Database Integration**: Store feedback and predictions in a database (PostgreSQL) for deeper analytics.
-   **User Interface**: Build a simple React/Streamlit frontend for manual testing.
