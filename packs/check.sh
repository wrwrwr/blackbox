#!/usr/bin/env bash

# Directly tests if a pack can be extracted and run.
tar -xzf $1_$2.tar.gz
rm -r ~/.pyxbld/*
python3 ./bot.py
rm -r ~/.pyxbld/*
