"""khatt — measure a logotype, understand it, cut it into parts, and set new Arabic words from those parts."""
from .pipeline import run
from .style import Style
from .typeset import typeset

__all__ = ["Style", "typeset", "run"]
