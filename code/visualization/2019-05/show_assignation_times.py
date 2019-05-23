import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import os
import re
from io import StringIO
from pandas.errors import EmptyDataError


if __name__ == "__main__":
    output_file_end_re = {
        "centroids": r"_centroids.npy",
        "results": r"_results.csv",
        "objective": r"_objective_.+.csv"
    }
