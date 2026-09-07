# Deck Vault — Stage 1 backend (complete files, no patch needed)

Extract this zip at the ROOT of your `backend` repo. It overwrites 5 files:

  requirements.txt
  routes/decks.py
  scripts/master_deck_pipeline.py
  services/deck_ai_gemini.py
  services/deck_ingestion_pipeline.py

Then DELETE these 2 dead files (a zip can't delete, so remove them by hand):

  git rm scripts/process_all_decks.py scripts/ingest_decks.py
  # or just delete them in your file explorer

Then:

  pip install -r requirements.txt          # pulls google-generativeai
  python -m unittest tests.test_deck_pipeline -v   # expect 8 passing

Enrich your decks (your GEMINI_API_KEY + Supabase/Drive env):

  python scripts/master_deck_pipeline.py --audit --dry-run   # eyeball first
  python scripts/master_deck_pipeline.py --audit             # real run
