# API Calls
# BoMeyering 2024

import requests
import numpy as np

# MARKER_ENDPOINT = "https://pgcview.org/api/v1/predict_markers"
# PGC_ENDPOINT = "https://pgcview.org/api/v1/predict_pgc"


MARKER_ENDPOINT = "http://localhost:8000/api/v1/predict_markers"
PGC_ENDPOINT = "http://localhost:8000/api/v1/predict_pgc"


def invoke_endpoints(path, **kwargs):
    """
    Invoke the marker and pgc prediction endpoints and return the results
    """
    # Make the marker model request
    files_body = {'file': open(path, 'rb')}
    marker_response = requests.post(MARKER_ENDPOINT, files=files_body, **kwargs)

    # Reopen img as binary
    files_body = {'file': open(path, 'rb')}
    pgc_response = requests.post(PGC_ENDPOINT, files=files_body, **kwargs)

    # Format responses
    filename = marker_response.json()['filename']
    data = marker_response.json()['data']

    marker_data = {
        'coordinates': np.array(marker_response.json()['data']['coordinates']),
        'classes': np.array(marker_response.json()['data']['classes'])
    }
    pgc_data = np.array(pgc_response.json()['data'])

    return filename, marker_data, pgc_data