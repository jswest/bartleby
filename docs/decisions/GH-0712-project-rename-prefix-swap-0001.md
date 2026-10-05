# `project rename` swaps the archive prefix on a raw connection (issue #712)

> Source: [#712](https://github.com/jswest/bartleby/issues/712)

`documents.file_path` and `images.file_path` store absolute archive paths that
embed the project name, so renaming a project means moving its directory and then
rewriting those paths. `rename_project` does the move first, then rewrites every
path that starts with `<projects>/<old>/archive/` to `<projects>/<new>/archive/` in
one transaction. If the rewrite fails, it moves the directory back. Three choices
here aren't obvious. (1) The prefix match uses `substr(file_path, 1, n) = ?`, not
`LIKE`, because project names can contain `_`, which `LIKE` treats as a wildcard.
The prefix also ends in a separator, so `archive2/…` can't match. Rows outside the
project's own archive are left alone. (2) The DB is opened with a raw
`apsw.Connection` + `_attach`, the same way `share/import_.py` does it, not with
`open_db`. `open_db` refuses a DB whose stamped schema version doesn't match, but
the rewrite touches only two long-standing columns. That means a stale corpus can
still be renamed and then upgraded or re-ingested under its new name. (3) A project
directory that has no `bartleby.db` is renamed with no rewrite. Making `file_path`
archive-relative, which would turn rename into a plain `mv`, is out of scope as
the issue says.
