import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configure API base URL from environment variable or use default
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8080")

def _extract_relevant_snippet(full_text: str, query_term: str, max_len: int = 250, padding: int = 70) -> str:
    """
    Extracts a snippet from full_text centered around the query_term, with a max length.
    """
    if not query_term or not full_text:
        if len(full_text) > max_len:
            return full_text[:max_len - 3] + "..."
        return full_text

    if len(full_text) <= max_len:
        return full_text

    lower_full_text = full_text.lower()
    lower_query_term = query_term.lower()
    query_len = len(lower_query_term)

    try:
        match_start_idx = lower_full_text.index(lower_query_term)
    except ValueError:  # Query not found
        return full_text[:max_len - 3] + "..."
    
    match_end_idx = match_start_idx + query_len

    # Initial window based on padding
    window_start = max(0, match_start_idx - padding)
    window_end = min(len(full_text), match_end_idx + padding)
    
    snippet_text = full_text[window_start:window_end]
    
    prefix = "..." if window_start > 0 else ""
    suffix = "..." if window_end < len(full_text) else ""
    
    current_result = prefix + snippet_text + suffix

    if len(current_result) > max_len:
        # Result is too long, need to trim snippet_text
        needed_len_for_snippet_text = max_len - (len(prefix) + len(suffix))
        
        if needed_len_for_snippet_text < 0: # Should not happen if max_len is reasonable
            needed_len_for_snippet_text = 0

        # Position of query within the current snippet_text
        query_start_in_snippet = match_start_idx - window_start
        # query_end_in_snippet = query_start_in_snippet + query_len (not directly used next)

        if query_len >= needed_len_for_snippet_text:
            # Query itself (or part of it) is all that can be shown in snippet_text
            # Show the beginning of the query term from its occurrence in snippet_text
            snippet_text = snippet_text[query_start_in_snippet : query_start_in_snippet + needed_len_for_snippet_text]
        else:
            # Query fits, center it within the new snippet_text length
            space_for_context = needed_len_for_snippet_text - query_len
            context_before_query = space_for_context // 2
            # context_after_query = space_for_context - context_before_query # implicit

            # Calculate new start for snippet_text, relative to current snippet_text's coordinate system
            new_snippet_text_start_rel = max(0, query_start_in_snippet - context_before_query)
            new_snippet_text_end_rel = new_snippet_text_start_rel + needed_len_for_snippet_text
            
            if new_snippet_text_end_rel > len(snippet_text):
                # If it overflows, shift window to the left
                new_snippet_text_end_rel = len(snippet_text)
                new_snippet_text_start_rel = max(0, new_snippet_text_end_rel - needed_len_for_snippet_text)
            
            snippet_text = snippet_text[new_snippet_text_start_rel:new_snippet_text_end_rel]
        
        current_result = prefix + snippet_text + suffix
        
        # Final safety net: if still too long (e.g. max_len extremely small)
        if len(current_result) > max_len:
            # Fallback to a hard truncation, prioritizing start of the (potentially shortened) result
            # This might truncate the query if max_len is pathologically small (e.g. < 10)
            return current_result[:max_len - 3] + "..."
            
    return current_result

@app.route('/')
def index():
    """Render the main search page."""
    return render_template('index.html')

@app.route('/search_text_action', methods=['POST'])
def search_text_action():
    """Handle text search requests."""
    # Note: These are not used when fetching JSON, kept for potential future use or clarity
    # query = request.form.get('query')
    # max_results = request.form.get('max_results', 20, type=int)

    # Note: Get data from request.get_json() as we are sending JSON via fetch
    data = request.get_json()
    if not data or 'query' not in data or not data['query']:
        return jsonify({"error": "Query content must be provided."}), 400

    query = data['query']
    # Get max_results from JSON data, provide default value
    max_results = data.get('max_results', 20)
    # Get search_type from JSON data, default to 'all'
    search_type = data.get('search_type', 'all')

    api_url = f"{API_BASE_URL}/search"
    # Update API request payload to include options
    payload = {
        "query": query,
        "max_results": max_results,
        "options": {
            "search_type": search_type,
            "refine_query": False,
            "rerank": True,
            "vector_weight": 0.9,
            "bm25_weight": 0.1
            # Other options can be added here if needed, e.g., refine_query, rerank
        }
    }

    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        api_response_data = response.json()

        # Extract relevant snippets for text results
        if 'results' in api_response_data and isinstance(api_response_data['results'], list):
            for item in api_response_data['results']:
                if 'text' in item and isinstance(item['text'], str) and item['text']:
                    item['text'] = _extract_relevant_snippet(item['text'], query)
        
        return jsonify(api_response_data)
    except requests.exceptions.HTTPError as e:
        try:
            error_details = e.response.json()
        except ValueError: # Handle cases where the error response is not JSON
            error_details = {"error": str(e)}
        print(f"API call HTTP error (Text Search): {e.response.status_code} - {error_details}")
        # Return JSON error and corresponding status code
        return jsonify(error_details), e.response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error calling Text Search API: {e}")
        return jsonify({"error": f"Error calling API: {e}"}), 500
    except Exception as e:
        print(f"Unexpected error during text search: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route('/search_image_action', methods=['POST'])
def search_image_action():
    """Handle image search requests."""
    # Note: image_file and max_results are now sent via FormData
    image_file = request.files.get('image_file')
    # Get max_results from form data
    max_results = request.form.get('max_results', 20, type=int)

    if not image_file or image_file.filename == '':
        return jsonify({"error": "An image file must be uploaded."}), 400

    api_url = f"{API_BASE_URL}/search_image"
    # Prepare files and data to send to the API
    files = {'image_file': (image_file.filename, image_file.stream, image_file.mimetype)}
    data = {'max_results': max_results}

    try:
        response = requests.post(api_url, files=files, data=data, timeout=60)
        response.raise_for_status()
        # Return the API's JSON response directly
        return jsonify(response.json())
    except requests.exceptions.HTTPError as e:
        try:
            error_details = e.response.json()
        except ValueError: # Handle cases where the error response is not JSON
            error_details = {"error": str(e)}
        print(f"API call HTTP error (Image Search): {e.response.status_code} - {error_details}")
        # Return JSON error and corresponding status code
        return jsonify(error_details), e.response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error calling Image Search API: {e}")
        return jsonify({"error": f"Error calling API: {e}"}), 500
    except Exception as e:
        print(f"Unexpected error during image search: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


if __name__ == '__main__':
    # Ensure Flask runs on a different port than the API
    print("Using api:", API_BASE_URL)
    app.run(host='0.0.0.0', port=5001)
