FROM python:3.11-slim

ENV GIT_PYTHON_REFRESH=quiet

RUN pip install poetry==1.6.1

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./start.sh ./supervisord.conf ./pyproject.toml ./README.md ./poetry.lock* ./

COPY ./package[s] ./packages

RUN poetry install  --no-interaction --no-ansi --no-root

COPY ./app ./app

RUN poetry install --no-interaction --no-ansi

EXPOSE 8001

CMD exec uvicorn app.server:gptweb --host 0.0.0.0 --port 8001
