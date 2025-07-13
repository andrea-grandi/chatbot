FROM python:3.10-slim

WORKDIR /app

RUN apt update && apt upgrade -y
RUN apt install git -y 
RUN pip install uv
RUN git clone https://github.com/andrea-grandi/chatbot && cd chatbot && uv sync

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
