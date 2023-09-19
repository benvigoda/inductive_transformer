import os.path
import csv
import typing
from googleapiclient.discovery import build  # type: ignore
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore
from google.auth.transport.requests import Request  # type: ignore
from google.oauth2.credentials import Credentials  # type: ignore
from pathlib import Path

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


SHEET_NAME = "Sheet1"


class Sheet:
    def __init__(
        self, path_to_api_creds=Path("google_sheets_api"), sheet_name=SHEET_NAME
    ):
        creds = None
        if os.path.exists(path_to_api_creds / "token.json"):
            creds = Credentials.from_authorized_user_file(
                path_to_api_creds / "token.json", SCOPES
            )
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    path_to_api_creds / "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=8001)
            # Save the credentials for the next run
            with open(path_to_api_creds / "token.json", "w") as token:
                token.write(creds.to_json())

        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        self.sheet = service.spreadsheets()
        self.current_input_row = 2
        self.debug = False
        self.sheet_name = sheet_name

    def ensure_sheets(self, spreadsheet_id, sheets):
        existing_sheets = [
            s.get("properties", {}).get("title")
            for s in self.sheet.get(spreadsheetId=spreadsheet_id)
            .execute()
            .get("sheets", [])
        ]
        sheets_to_create = list(set(sheets) - set(existing_sheets))
        if not sheets_to_create:
            return
        add_sheets_request = {
            "requests": [
                {
                    "addSheet": {"properties": {"title": title}},
                }
                for title in sheets_to_create
            ],
        }
        self.sheet.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=add_sheets_request,
        ).execute()

    def update_table(self, spreadsheet_id, sheet_name, table):
        self.sheet.values().clear(
            spreadsheetId=spreadsheet_id, range=sheet_name
        ).execute()
        self.sheet.values().update(
            spreadsheetId=spreadsheet_id,
            range=sheet_name,
            valueInputOption="RAW",
            body=dict(values=table),
        ).execute()

    def save_sheet(self, spreadsheetId, range, path):
        result = (
            self.sheet.values().get(spreadsheetId=spreadsheetId, range=range).execute()
        )
        values = iter(result.get("values", []))
        with open(path, "w") as f:
            write = csv.writer(f)
            write.writerows(values)

    def is_empty(self, spreadsheetId, range) -> bool:
        result = (
            self.sheet.values().get(spreadsheetId=spreadsheetId, range=range).execute()
        )
        values = list(result.get("values", {}))
        if values:
            return False
        else:
            return True

    def update_column_width(self, spreadsheet_id, sheets, end=1):
        sheet_name_id = self.get_sheetId_by_sheetName(spreadsheet_id, sheets)
        update_column_width_request = {
            "requests": [
                {
                    "autoResizeDimensions": {
                        "dimensions": {
                            "sheetId": sheet_name_id[title],
                            "dimension": "COLUMNS",
                            "startIndex": 0,
                            "endIndex": end,
                        }
                    }
                }
                for title in sheets
            ],
        }
        self.sheet.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=update_column_width_request,
        ).execute()

    def update_cells_color(
        self,
        spreadsheet_id,
        sheet_name,
        row_start=0,
        row_end=1,
        bold=True,
        backgroundColor=[0.8, 0.8, 0.8],
        frozen_row=0,
    ):
        sheet_name_id = self.get_sheetId_by_sheetName(spreadsheet_id, [sheet_name])
        body_request = {
            "requests": [
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_name_id[sheet_name],
                            "startRowIndex": row_start,
                            "endRowIndex": row_end,
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": {
                                    "red": backgroundColor[0],
                                    "green": backgroundColor[1],
                                    "blue": backgroundColor[2],
                                },
                                "textFormat": {
                                    "foregroundColor": {
                                        "red": 0.0,
                                        "green": 0.0,
                                        "blue": 0.0,
                                    },
                                    "fontSize": 10,
                                    "bold": bold,
                                },
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                    }
                },
                {
                    "updateSheetProperties": {
                        "properties": {
                            "sheetId": sheet_name_id[sheet_name],
                            "gridProperties": {"frozenRowCount": frozen_row},
                        },
                        "fields": "gridProperties.frozenRowCount",
                    },
                },
            ]
        }
        self.sheet.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body_request,
        ).execute()

    def get_sheetId_by_sheetName(self, spreadsheet_id, sheets) -> typing.Dict[str, int]:
        spreadsheet = self.sheet.get(spreadsheetId=spreadsheet_id).execute()
        sheet_name_id: typing.Dict[str, int] = {}
        for sheet_name in sheets:
            sheet_id = None
            for _sheet in spreadsheet["sheets"]:
                if _sheet["properties"]["title"] == sheet_name:
                    sheet_id = _sheet["properties"]["sheetId"]
                    sheet_name_id[sheet_name] = sheet_id
        return sheet_name_id
