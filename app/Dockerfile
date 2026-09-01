FROM python:3.11-slim

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "pydantic>=2.0" \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    lightgbm \
    pyarrow

WORKDIR /workspace

COPY app/ ./app/

# Data artifacts and model are mounted at runtime — not baked into the image.
# docker run -v $(pwd)/data:/workspace/data -v $(pwd)/outputs:/workspace/outputs ...

ENV DATA_DIR=/workspace/data
ENV MODEL_PATH=/workspace/outputs/runs/latest/model.joblib

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
