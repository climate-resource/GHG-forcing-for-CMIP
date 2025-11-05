#!/bin/bash
printf "url: %s\nkey: %s\n" "${ECMWF_DATASTORES_URL}" "${ECMWF_DATASTORES_KEY}" > "$HOME/.ecmwfdatastoresrc"
#printf "url: %s\nkey: %s\n" "${ECMWF_DATASTORES_URL}" "test" >"$HOME/.ecmwfdatastoresrc"
