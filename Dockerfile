FROM python:3.12-slim
USER root

RUN apt-get update && apt-get install -y libmagic1

RUN mkdir -p /knowledge-graphs
WORKDIR /knowledge-graphs

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY pgs ./pgs
COPY app.py ./app.py
COPY config_example.env ./config_example.env

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "server.address=0.0.0.0"]