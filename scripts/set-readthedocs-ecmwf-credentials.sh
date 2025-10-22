#!/bin/bash
printf "url: %s\nkey: %s\n" "${ECMWF_DATASTORES_URL}" "${ECMFW_DATASTORES_KEY}" >"$HOME/.ecmwfdatastoresrc"
