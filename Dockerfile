FROM python:3.10-slim

WORKDIR /app

RUN pip install uv
RUN uv init
RUN uv add fastapi openai python-dotenv uvicorn 

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
