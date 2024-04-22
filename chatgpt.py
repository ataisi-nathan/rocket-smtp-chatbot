from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import const
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

app = Flask(__name__)
CORS(app)

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = const.APIKEY

@app.route('/query', methods=['GET'])
def handle_query():
    try:
        query = request.args.get('q', '')

        if not query:
            return jsonify({'error': 'Query parameter is missing'}), 400

        # Load the text document
        loader = TextLoader('data.txt')
        index = VectorstoreIndexCreator().from_loaders([loader])

        # Query the index
        response = index.query(query)
        
        return jsonify({'response': response})
    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        # Return an error response
        return jsonify({'error': 'An error occurred while processing the request'}), 500

if __name__ == '__main__':
    app.run(debug=True)
