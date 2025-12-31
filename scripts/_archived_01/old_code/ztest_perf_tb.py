from halib import *
from halib.research.perftb import PerfTB
from halib import *
from argparse import ArgumentParser


def main():
    tb = PerfTB.from_csv("./performance_table.csv")
    tb.display()
    tb.plot(save_path="./a.svg", title="Performance Table Plot", open_plot=True)


if __name__ == "__main__":
    main()
