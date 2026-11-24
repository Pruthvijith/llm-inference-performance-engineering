FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt constraints-cpu.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 && pip install --no-cache-dir -c constraints-cpu.txt -r requirements.txt
COPY serving ./serving
RUN useradd --uid 10001 --create-home inference
USER 10001
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "serving.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
