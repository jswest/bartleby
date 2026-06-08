# `search` triage signals

each hit carries `rank` (1-indexed) and `normalized_score` (top hit = 1.0). Raw RRF `score` is tiny by design (~`0.015–0.033`) and only comparable within one query.
