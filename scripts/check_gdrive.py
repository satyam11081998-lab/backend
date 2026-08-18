import os
import dotenv
from services import gdrive

dotenv.load_dotenv(override=True)
print("is_configured:", gdrive.is_configured())
print("folder_id:", bool(gdrive._folder_id()))
print("refresh_token:", bool(os.environ.get("GOOGLE_DRIVE_REFRESH_TOKEN")))
print("client_id:", bool(os.environ.get("GOOGLE_DRIVE_CLIENT_ID")))
print("client_secret:", bool(os.environ.get("GOOGLE_DRIVE_CLIENT_SECRET")))