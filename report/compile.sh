#!/bin/bash

platex main.tex
bibtex main
platex main.tex
platex main.tex
dvipdfmx -f font.map main.dvi
