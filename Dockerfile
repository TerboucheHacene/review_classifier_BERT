FROM python:3.8.5-slim


ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.6


RUN mkdir MyApp
WORKDIR MyApp
COPY pyproject.toml pyproject.toml
COPY app app
COPY review_cl review_cl
COPY logs logs

ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip install "poetry==$POETRY_VERSION"

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev
#RUN poetry shell

EXPOSE 5000

CMD ["uvicorn", "app.api:app", "--host" , "0.0.0.0", "--port",  "5000"]