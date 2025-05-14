# Use the official torch image as a base image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get -y update 
RUN apt-get install -y --no-install-recommends libgl1-mesa-dev libglib2.0-0
RUN rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "server.py"] 
