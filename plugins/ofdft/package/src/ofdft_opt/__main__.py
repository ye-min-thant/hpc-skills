"""python -m ofdft_opt <command> ...   (also: python -m mpi4py -m ofdft_opt <command> ...)"""
import sys

from ofdft_opt.cli import main

if __name__ == "__main__":
    sys.exit(main())
