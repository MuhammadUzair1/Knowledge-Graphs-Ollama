FROM python:3.12
USER root

RUN mkdir -p /knowledge-graphs
WORKDIR /knowledge-graphs

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY pgs ./pgs
COPY app.py ./app.py
COPY config.env ./config.env

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "server.address=0.0.0.0"]