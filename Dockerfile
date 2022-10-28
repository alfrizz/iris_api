
FROM python:3.10-slim-buster

COPY iris_requirements.txt .
RUN pip install -r iris_requirements.txt

COPY ./app .

EXPOSE 7000

CMD ["uvicorn", "iris_main:app", "--host", "0.0.0.0", "--port", "7000"]




