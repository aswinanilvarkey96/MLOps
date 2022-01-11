#!/bin/sh
# Decrypt the file
# --batch to prevent interactive command
# --yes to assume "yes" for questionsssss
pwd
gpg --quiet --batch --yes --decrypt --passphrase="$SECRET" --output "$pwd.github/scripts/secret.json" "$pwd/.github/scripts/secret.json.gpg"