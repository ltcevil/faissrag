#!/bin/bash
#supervisord -c /code/supervisord.conf
# Start the Python application in the background
uvicorn app.server:app --host 0.0.0.0 --port 8001 &

# Start the Node.js application
npm run start:server:prod
