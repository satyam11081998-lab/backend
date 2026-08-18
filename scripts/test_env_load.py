import os
import dotenv

env_path = os.path.abspath(".env")
dotenv.load_dotenv(env_path, override=True)

from services import gdrive
print("env_path:", env_path)
print("is_configured:", gdrive.is_configured())
print("folder_id:", gdrive._folder_id())
print("refresh_token set:", bool(os.environ.get("GOOGLE_DRIVE_REFRESH_TOKEN")))