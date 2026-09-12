"""agentic-ml: an agentic framework for machine learning tasks.

The base import intentionally stays light (no LLM / Strands dependency) so that the exported
:class:`~agentic_ml.core.model.AgenticModel` can be deployed to production on its own. To run
research, import from :mod:`agentic_ml.estimators` with the ``research`` extra installed.
"""

from agentic_ml.core.model import AgenticModel

__version__ = "0.1.0"

__all__ = ["AgenticModel", "__version__"]
