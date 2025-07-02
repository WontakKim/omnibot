#!/bin/bash

create_vm() {
  if ! docker images android-local -q | grep -q .; then
    echo "Image not found locally. Building..."
    docker build -t android-local ..
  else
    echo "Image found locally. Skipping build."
  fi

  docker compose -f ../compose.yml up -d

  # Wait for the Emulator to start up
  while true; do
    status=$(docker exec -it client-android cat device_status 2>/dev/null)
    if [ "$status" = "READY" ]; then
      break
    fi
    echo "Waiting for a response from the docker emulator."
    sleep 5
  done

  # # Wait for the VM to start up
  # while true; do
  #   response=$(curl --write-out '%{http_code}' --silent --output /dev/null localhost:4723/status)
  #   if [ "$response" -eq 200 ]; then
  #     break
  #   fi
  #     echo "Waiting for a response from the computer control server. When first building the VM storage folder this can take a while..."
  #     sleep 5
  # done

  echo "VM + server is up and running!"
}

# Check if control parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 [create|start|stop|delete]"
  exit 1
fi

# Execute the appropriate function based on the control parameter
case "$1" in
  "create")
    create_vm
    ;;
  "start")
    start_vm
    ;;
  "stop")
    stop_vm
    ;;
  "delete")
    delete_vm
    ;;
  *)
    echo "Invalid option: $1"
    echo "Usage: $0 [create|start|stop|delete]"
    exit 1
    ;;
esac