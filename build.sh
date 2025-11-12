#!env bash
run() { echo "$*"; "$@"; }
run g++ -std=c++20 main.cpp -o main
