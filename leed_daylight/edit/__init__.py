# leed_daylight/edit/__init__.py
"""Safe edits of a project's input files.

Moved from the browser wizard (``drivers/wizard_app``) when the wizard was
left out of this copy: ``files`` writes a file through ``.part``, backs the
old one up to ``_run/backup/`` and refuses links; ``project_yaml`` writes
project.yaml in the template's layout. Neither serves a page or starts a
process. See PORTED.md, section "Copied into leed-mcp".
"""
