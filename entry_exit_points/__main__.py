"""Allow running as: python -m entry_exit_points"""

from .cli import main

raise SystemExit(main())
