import os
import dotenv

for env_file in ["C:/Users/satya/Videos/company/consilio/.env.local", "C:/Users/satya/Videos/company/consilio/.env"]:
    if os.path.exists(env_file):
        vals = dotenv.dotenv_values(env_file)
        print(f"Reading {env_file}, keys found: {len(vals)}")
        gdrive_keys = {k: v for k, v in vals.items() if any(x in k for x in ["GDRIVE", "GOOGLE"])}
        print(f"Gdrive keys in {env_file}: {list(gdrive_keys.keys())}")
        if gdrive_keys:
            backend_env = "C:/Users/satya/Videos/company/consilio-backend/.env"
            with open(backend_env, "a", encoding="utf-8") as f:
                f.write("\n# Google Drive Config\n")
                for k, v in gdrive_keys.items():
                    if v:
                        f.write(f'{k}="{v}"\n')
            print(f"Successfully copied {list(gdrive_keys.keys())} to backend .env")
            break