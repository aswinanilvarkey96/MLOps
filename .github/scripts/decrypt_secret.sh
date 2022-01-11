#!/bin/sh
# Decrypt the file
# --batch to prevent interactive command
# --yes to assume "yes" for questionsss
pwd
gpg --quiet --batch --yes --decrypt --passphrase="$SECRET" --output secret.json secret.json.gpg