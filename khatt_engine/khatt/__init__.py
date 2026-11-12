"""khatt — analyse a logotype image and set new Arabic words in the same style."""
from .analyze import analyze_image
from .pipeline import generate, run
from .style import Style
from .typeset import typeset

__all__ = ["Style", "analyze_image", "typeset", "generate", "run"]
