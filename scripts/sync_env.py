import dotenv

c_env = dotenv.dotenv_values("../consilio/.env.local")
if not c_env:
    c_env = dotenv.dotenv_values("../consilio/.env")

b_path = ".env"
existing = dotenv.dotenv_values(b_path)

to_add = {}
for k in [
    "GDRIVE_FOLDER_ID", "GDRIVE_SUBMISSIONS_FOLDER_ID",
    "GOOGLE_DRIVE_CLIENT_ID", "GOOGLE_DRIVE_CLIENT_SECRET", "GOOGLE_DRIVE_REFRESH_TOKEN",
    "GOOGLE_SA_CREDENTIALS", "GOOGLE_SA_CLIENT_EMAIL", "GOOGLE_SA_PRIVATE_KEY"
]:
    if k in c_env and k not in existing and c_env[k]:
        to_add[k] = c_env[k]

if to_add:
    with open(b_path, "a", encoding="utf-8") as f:
        f.write("\n# Google Drive credentials copied from frontend\n")
        for k, v in to_add.items():
            f.write(f'{k}="{v}"\n')
    print(f"Added {list(to_add.keys())} to {b_path}")
else:
    print("No missing keys to add")