#!/bin/bash


. $(poetry env info --path)/bin/activate

uvicorn app.api:app \
    --host 0.0.0.0 \
    --port 5000 \
    --reload \
    --reload-dir review_cl \
    --reload-dir app 
      