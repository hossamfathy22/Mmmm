"""The benchmark: the engine examined against things whose right answer is known.

Open-licensed typefaces are the ground truth. A word set in a typeface is a logo whose every
measurement and every other letter we already know:

  measuring  the same word is turned, slanted, shrunk, outlined, set on an uneven ground or beside
             a symbol, each by a known amount, and the engine must report the same numbers;
  letters    the engine reads the word, draws the letters it was not shown, and each is compared
             with the typeface's own letter.

`python -m khatt.bench` runs both and prints a report. The typefaces are downloaded once into a
local cache (they are not part of this project); fontTools and uharfbuzz are needed for this only.
"""
