#!/bin/bash

export NODE_ENV=development

# Start the webpack devserver.
npx webpack server &
pid[1]=$!

# Run the node server.
poetry run server
pid[0]=$!

# When control+c is pressed, kill all process ids.
trap "kill ${pid[0]} ${pid[1]}; exit 1" INT
wait

