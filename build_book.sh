#!/usr/bin/env bash

rm -r docs/*
touch docs/.nojekyll
jupyter-book build jupyter_book/
 cp -r jupyter_book/_build/html/* docs/
