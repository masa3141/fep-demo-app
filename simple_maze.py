import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import streamlit as st
from utils import *


def main():
    """Create  the grid locations in the form of a list of (Y, X) tuples -- HINT: use itertools"""
    grid_locations = list(itertools.product(range(3), repeat=2))
    print(grid_locations)
    plot_grid(grid_locations)
