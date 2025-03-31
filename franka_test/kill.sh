#!/usr/bin/env bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color


kill -9 $(pidof python)
kill $(lsof -ti tcp -c ^code -c ^firefox -c ^chrome) 
# note: can exclude command from tcp with `-c ^CMD` e.g. `-c ^code` excludes code
# can also restrict to only programs you care about with `-c *CMD` e.g. `-c *ros`

# rosnode cleanup & echo "y"

lsof -i tcp -c ^code -c ^firefox -c ^chrome

echo -e "${RED}kill any remining processes listed above with 'kill -9 <PID>'${NC}
"
