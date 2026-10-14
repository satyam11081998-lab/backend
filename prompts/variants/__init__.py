"""Frozen prompt-stack variants for the admin interviewer arena.

A VARIANT is one implementation of the interviewer prompt layer, behind the single
seam the engine already uses: build_adaptive_interviewer_messages(). Everything
else -- session signals, gates, response function, control packet -- stays shared,
because that is the part the arena is NOT testing.

Why frozen snapshots rather than git branches: two branches cannot be imported into
one process, and a vote recorded against "the branch" stops meaning anything the
moment that branch moves or merges. A snapshot is a measurement baseline.
"""
