#!/bin/sh

# Decrypt the file
# --batch to prevent interactive command
# --yes to assume "yes" for questionsss
gpg --quiet --batch --yes --decrypt --passphrase="$SECRET" \
--output application_default_credentials.json application_default_credentials.json.gpg