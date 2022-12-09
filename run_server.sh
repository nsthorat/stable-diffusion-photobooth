#!/bin/bash

export NODE_ENV=development

# Start the webpack devserver.
npx webpack server &
pid[1]=$!

# Run the node server.
poetry run python -m flask --app server/server run --host=0.0.0.0 --port=8000 --cert=adhoc
pid[0]=$!

# When control+c is pressed, kill all process ids.
trap "kill ${pid[0]} ${pid[1]}; exit 1" INT
wait

