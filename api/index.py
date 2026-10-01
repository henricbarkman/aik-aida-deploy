import logging
import os
import sys

# Vercel shows the function's stderr in its log. With no handler configured,
# Python prints only WARNING and up, bare (logging.lastResort), and every
# logger.info in the pipeline (baseline chunking, price hits, token fallback)
# is dropped. basicConfig does nothing if the runtime already set up a handler.
_level = getattr(logging, os.environ.get('AIDA_LOG_LEVEL', 'INFO').upper(), None)
logging.basicConfig(
    level=_level if isinstance(_level, int) else logging.INFO,
    stream=sys.stderr,
    format='%(levelname)s %(name)s: %(message)s',
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from aida.web.app import app  # noqa: E402,F401  (Vercel's WSGI entry point)
