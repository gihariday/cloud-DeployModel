FROM python:3.9

WORKDIR /code

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENV HOST 0.0.0.0

EXPOSE 8080

CMD ["python", "main.py"]