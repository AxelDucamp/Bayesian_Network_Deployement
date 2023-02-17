FROM python:3.9.1
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD gunicorn --workers=4 --bind 0.0.0.0:8080 app:app