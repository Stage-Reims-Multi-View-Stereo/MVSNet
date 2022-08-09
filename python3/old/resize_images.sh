#!/bin/bash

# Optional argument: directory where image to convert are
if [ -n "$1" ]; then
    cd "$1" || exit 1
fi

find . -iname '*.png' -exec convert -resize '1536x864!' {} {}_ \; -exec mv {}_ {} \;