# K · Google Drive storage architecture

## 1. Structure
```
<II_GDRIVE_ROOT_FOLDER_ID>  (a folder the storage account owns, or a Shared Drive folder)
└── Users
    ├── u_3f9a1c…   (opaque: HMAC-SHA256(II_DRIVE_FOLDER_SALT, user_id)[:16] — no names/emails)
    │   ├── CV          cv_<document_id>_<sha12>.pdf
    │   ├── JD          jd_<document_id>_<sha12>.docx | .txt
    │   └── Interviews  report_<session_id>.md   (flag drive.export_reports)
    └── u_…
```
* Folder names carry **no PII**; the mapping lives only in the II database.
* Folder ids are stored in `drive_folders (scope_key, kind)` with a unique constraint, so a
  user folder is created once. Before creating, the client searches the parent for the
  exact name (covers a crash between Drive create and DB write).
* Each uploaded file carries `appProperties {ii_doc_id, ii_sha256}`; before uploading the
  sync job queries by `ii_doc_id` → re-runs and retries never create duplicates.
* Drive ids never leave the backend (API responses expose only `storage_status`).

## 2. Flow
```
upload → validate → sha256 → dedupe (user, kind, sha256) → encrypted blob + text in II DB
       → documents.storage_status = pending → job drive_sync_document
job: ensure folders → find by appProperties → resumable upload → drive_files.synced
     (failure → attempts+1, exponential backoff 1m/5m/30m/2h/12h, max 6 → failed, admin retry)
```
The II database copy is the durable source until the Drive copy is confirmed, so a Drive
outage never loses a document. After a confirmed sync the encrypted blob may be purged
(`II_PURGE_BLOB_AFTER_DRIVE_SYNC`, default false — the parsed text is kept for re-analysis).

## 3. Auth options (same as the Deck Vault, separate env names)
`II_GDRIVE_REFRESH_TOKEN + II_GDRIVE_CLIENT_ID + II_GDRIVE_CLIENT_SECRET` (a dedicated
storage Google account), or a service account `II_GDRIVE_SA_JSON` (base64) writing into a
**Shared Drive** folder (service accounts have no My-Drive quota). Not configured →
`storage_status = disabled`, everything else works.

## 4. Security
Scope `drive.file` is enough when the app created the folders; `drive` is required only if
the root was created manually and shared. Files are never shared publicly; no
`webViewLink` is returned to clients. Deletion of a document or account enqueues
`drive_delete` jobs. All sync transitions are written to `audit_logs`.
