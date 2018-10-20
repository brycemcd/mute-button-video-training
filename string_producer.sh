#!/bin/bash

# ./string_producer.sh 100000 | kafka-console-producer.sh --broker-list 10.1.2.206 --topic test-kaf

STOP_COUNT=$1
COUNTER=0

while [  $COUNTER -lt $STOP_COUNT ]; do
  head /dev/urandom | tr -dc A-Za-z0-9 | head -n 10; echo '';
  let COUNTER=COUNTER+1
done
