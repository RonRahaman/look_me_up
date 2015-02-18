#!/usr/bin/env bash

mpiexec.openmpi -n 1 --bind-to-socket --rankfile 1rank_2socket.rankfile ./main_mpi_omp -t 16
mpiexec.openmpi -n 2 --bind-to-socket --rankfile 2rank_2socket.rankfile ./main_mpi_omp -t 8
