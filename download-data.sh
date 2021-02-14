#!/bin/bash

cd $(dirname $0) || exit $?

mkdir -p data && cd data || exit $?

if [[ ! -f "text8" ]]; then
    wget http://mattmahoney.net/dc/text8.zip -O text8.gz
    gzip -d text8.gz -f
fi
