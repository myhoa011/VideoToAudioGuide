FROM python:3.10.16-slim

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && apt-get clean

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "src/app.py"]