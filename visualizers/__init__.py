import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Set the font parameters for LaTeX-style rendering
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
# Use full LaTeX for all text (requires a LaTeX installation). If unavailable, comment out the next line.
# plt.rcParams["text.usetex"] = True
# LaTeX-style tick labels
latex_formatter = FuncFormatter(lambda v, pos: rf"${v:g}$")
