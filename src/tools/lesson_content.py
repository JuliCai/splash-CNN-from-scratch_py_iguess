"""
puts the lesson together for tools/build_notebooks.py.

the main lab's text lives in lesson_lab.py and the bonus notebooks' text in lesson_bonus.py. the code inside the
todo and given cells comes from babytorch.py and addons.py, so theres only one copy of every answer.
"""

from lesson_bonus import BONUS_NOTES, BONUS_STUBS, addon_a, addon_b
from lesson_lab import LAB_STUBS, NOTES as LAB_NOTES, core

STUBS = {**LAB_STUBS, **BONUS_STUBS}      # what a student sees in every todo cell
NOTES = {**LAB_NOTES, **BONUS_NOTES}      # the doc comments on top of every given code cell

NOTEBOOKS = {
    "lab.ipynb": core,
    "addon_A_batching.ipynb": addon_a,
    "addon_B_split_k.ipynb": addon_b,
}
