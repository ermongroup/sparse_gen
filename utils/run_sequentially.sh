#!/bin/bash

# Make sure everything in scripts is executable
chmod +x ./$1/*.sh

# Run files one by one
for filename in ./$1/*.sh; do
    bash $filename
done
