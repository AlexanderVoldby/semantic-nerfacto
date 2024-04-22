import io
import os
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload

class GDriveDownloader:
    def __init__(self, credentials_file='credentials.json', token_file='token.json', scopes=['https://www.googleapis.com/auth/drive']):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.scopes = scopes
        self.service = self.authenticate()

    def authenticate(self):
        creds = None
        # Check if the token file exists and load it
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
            except Exception as e:
                print(f"Error loading credentials: {e}")
                creds = None

        # If no valid credentials, try to refresh or generate new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except RefreshError:
                    creds = None  # Ensure flow attempts to get new credentials below
            if not creds:
                # Attempt to create new credentials
                if os.path.exists(self.credentials_file):
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, self.scopes)
                    creds = flow.run_local_server(port=0)
                    # Save the credentials for the next run
                    with open(self.token_file, 'w') as token:
                        token.write(creds.to_json())
                else:
                    raise FileNotFoundError(f"{self.credentials_file} does not exist. Please provide a valid credentials file.")

        return build('drive', 'v3', credentials=creds)

    def generate_new_token(self):
        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, self.scopes)
        creds = flow.run_local_server()  # Use run_console instead of run_local_server for non-interactive environments
        with open(self.token_file, 'w') as token:
            token.write(creds.to_json())
        print("New token generated and saved.")

    def download_file(self, file_id, file_name, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        request = self.service.files().get_media(fileId=file_id)
        file_path = os.path.join(folder_path, file_name)
        fh = io.FileIO(file_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        fh.close()
        return file_path

    def list_files_in_folder(self, folder_id):
        # First, get the name of the folder
        folder_info = self.service.files().get(fileId=folder_id, fields='name').execute()
        folder_name = folder_info.get('name', 'Unknown Folder')  # Extract the folder name

        # Then, list all files in the folder as before
        query = f"'{folder_id}' in parents"
        result = []
        page_token = None
        while True:
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()
            result.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if page_token is None:
                break

        # Return both the folder name and the list of files
        return folder_name, result

    def download_folder(self, folder_id, local_path):
        items = self.list_files_in_folder(folder_id)
        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                # Recursively download contents of subdirectories
                new_local_path = os.path.join(local_path, item['name'])
                self.download_folder(item['id'], new_local_path)
            else:
                # Download files directly within the folder
                self.download_file(item['id'], item['name'], local_path)